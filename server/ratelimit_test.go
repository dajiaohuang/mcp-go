package server

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/time/rate"
)

type rateLimitTestSession struct{ id string }

func (rateLimitTestSession) Initialize()                                         {}
func (rateLimitTestSession) Initialized() bool                                   { return true }
func (rateLimitTestSession) NotificationChannel() chan<- mcp.JSONRPCNotification { return nil }
func (s rateLimitTestSession) SessionID() string                                 { return s.id }

type rateLimitTestContextKey struct{}

func rateLimitTestOptions() RateLimitOpts {
	return RateLimitOpts{
		PerSessionRPS:   rate.Limit(0.001),
		PerSessionBurst: 1,
		ReapInterval:    -1,
	}
}

func invokeRateLimitHandler(t *testing.T, handler ToolHandlerFunc, ctx context.Context) *mcp.CallToolResult {
	t.Helper()
	result, err := handler(ctx, mcp.CallToolRequest{})
	require.NoError(t, err)
	return result
}

func TestRateLimitPerSession(t *testing.T) {
	var calls atomic.Int64
	handler := rateLimitMiddleware(rateLimitTestOptions())(func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		calls.Add(1)
		return mcp.NewToolResultText("called"), nil
	})
	ctxA := context.WithValue(context.Background(), clientSessionKey{}, rateLimitTestSession{id: "a"})
	ctxB := context.WithValue(context.Background(), clientSessionKey{}, rateLimitTestSession{id: "b"})

	assert.NotNil(t, invokeRateLimitHandler(t, handler, ctxA))
	denied := invokeRateLimitHandler(t, handler, ctxA)
	require.NotNil(t, denied) // denied after the burst
	assert.True(t, denied.IsError)
	assert.NotNil(t, invokeRateLimitHandler(t, handler, ctxB)) // independent session bucket
	assert.EqualValues(t, 2, calls.Load())
}

func TestRateLimitCallsWithoutSessionShareBucket(t *testing.T) {
	var calls atomic.Int64
	handler := rateLimitMiddleware(rateLimitTestOptions())(func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		calls.Add(1)
		return mcp.NewToolResultText("called"), nil
	})

	assert.NotNil(t, invokeRateLimitHandler(t, handler, context.Background()))
	assert.NotNil(t, invokeRateLimitHandler(t, handler, context.Background()))
	assert.EqualValues(t, 1, calls.Load())
}

func TestRateLimitGlobalCeiling(t *testing.T) {
	opts := rateLimitTestOptions()
	opts.PerSessionBurst = 2
	opts.GlobalRPS = rate.Limit(0.001)
	opts.GlobalBurst = 1
	var calls atomic.Int64
	handler := rateLimitMiddleware(opts)(func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		calls.Add(1)
		return mcp.NewToolResultText("called"), nil
	})
	ctxA := context.WithValue(context.Background(), clientSessionKey{}, rateLimitTestSession{id: "a"})
	ctxB := context.WithValue(context.Background(), clientSessionKey{}, rateLimitTestSession{id: "b"})

	assert.NotNil(t, invokeRateLimitHandler(t, handler, ctxA))
	denied := invokeRateLimitHandler(t, handler, ctxB) // denied by the shared ceiling
	require.NotNil(t, denied)
	assert.True(t, denied.IsError)
	assert.EqualValues(t, 1, calls.Load())
}

func TestRateLimitCustomKeyAndDenial(t *testing.T) {
	opts := rateLimitTestOptions()
	opts.KeyFunc = func(ctx context.Context) string {
		return ctx.Value(rateLimitTestContextKey{}).(string)
	}
	denied := false
	opts.OnDeny = func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		denied = true
		return mcp.NewToolResultError("custom denial"), nil
	}
	var calls atomic.Int64
	handler := rateLimitMiddleware(opts)(func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		calls.Add(1)
		return mcp.NewToolResultText("called"), nil
	})
	ctx := context.WithValue(context.Background(), rateLimitTestContextKey{}, "tenant")

	assert.NotNil(t, invokeRateLimitHandler(t, handler, ctx))
	assert.NotNil(t, invokeRateLimitHandler(t, handler, ctx))
	assert.True(t, denied)
	assert.EqualValues(t, 1, calls.Load())
}

func TestRateLimitInvalidOptions(t *testing.T) {
	tests := []struct {
		name string
		edit func(*RateLimitOpts)
	}{
		{name: "zero per-session rate", edit: func(opts *RateLimitOpts) { opts.PerSessionRPS = 0 }},
		{name: "not-a-number per-session rate", edit: func(opts *RateLimitOpts) { opts.PerSessionRPS = rate.Limit(math.NaN()) }},
		{name: "zero burst", edit: func(opts *RateLimitOpts) { opts.PerSessionBurst = 0 }},
		{name: "half global configuration", edit: func(opts *RateLimitOpts) { opts.GlobalRPS = rate.Limit(1) }},
		{name: "negative stale TTL", edit: func(opts *RateLimitOpts) { opts.StaleTTL = -time.Second }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			opts := rateLimitTestOptions()
			test.edit(&opts)
			assert.Panics(t, func() { newRateLimitStore(opts) })
		})
	}
}

func TestRateLimitRemovesIdleEntries(t *testing.T) {
	opts := rateLimitTestOptions()
	opts.StaleTTL = time.Minute
	store := newRateLimitStore(opts)
	require.True(t, store.allow(context.Background()))
	assert.False(t, store.allow(context.Background()))

	store.mu.Lock()
	store.entries[""].lastSeen = time.Now().Add(-2 * time.Minute)
	store.mu.Unlock()

	assert.True(t, store.removeStaleEntries(time.Now()))
	assert.Empty(t, store.entries)
	assert.True(t, store.allow(context.Background()))
	assert.False(t, store.allow(context.Background()))
}

func TestRateLimitConcurrentSessions(t *testing.T) {
	opts := rateLimitTestOptions()
	opts.PerSessionBurst = 1
	opts.KeyFunc = func(ctx context.Context) string { return ctx.Value(rateLimitTestContextKey{}).(string) }
	var calls atomic.Int64
	handler := rateLimitMiddleware(opts)(func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		calls.Add(1)
		return mcp.NewToolResultText("called"), nil
	})

	const sessions = 24
	var wg sync.WaitGroup
	for i := range sessions {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx := context.WithValue(context.Background(), rateLimitTestContextKey{}, string(rune('a'+i)))
			for range 3 {
				_, err := handler(ctx, mcp.CallToolRequest{})
				if err != nil {
					calls.Add(1000)
					return
				}
			}
		}()
	}
	wg.Wait()
	assert.EqualValues(t, sessions, calls.Load())
}
