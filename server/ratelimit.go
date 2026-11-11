package server

import (
	"context"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"golang.org/x/time/rate"
)

const (
	defaultRateLimitReapInterval = 5 * time.Minute
	defaultRateLimitStaleTTL     = 10 * time.Minute
)

// RateLimitOpts configures the per-session tool-call rate limiter.
type RateLimitOpts struct {
	// PerSessionRPS is the sustained rate for each session. It must be positive.
	PerSessionRPS rate.Limit
	// PerSessionBurst is the maximum burst allowed for each session. It must be positive.
	PerSessionBurst int

	// GlobalRPS and GlobalBurst optionally add a server-wide ceiling. Set both to
	// positive values to enable it; leave both zero to disable it.
	GlobalRPS   rate.Limit
	GlobalBurst int

	// ReapInterval controls how often idle session limiters are removed. Zero
	// uses the default interval; a negative value disables idle-entry cleanup.
	ReapInterval time.Duration
	// StaleTTL is how long a session limiter is retained after its last call.
	// Zero uses the default TTL; negative values are invalid.
	StaleTTL time.Duration

	// KeyFunc overrides the default session-ID key. Without it, calls with no
	// client session share one limiter. Empty keys also share one limiter.
	KeyFunc func(context.Context) string
	// OnDeny customizes the tool result returned for a limited call. When nil,
	// the middleware returns an IsError tool result.
	OnDeny func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error)
	// Logger receives a debug message when a tool call is denied.
	Logger *slog.Logger
}

type rateLimitEntry struct {
	limiter  *rate.Limiter
	lastSeen time.Time
	active   int
}

type rateLimitStore struct {
	mu sync.Mutex // protects entries and reaperCancel

	entries      map[string]*rateLimitEntry
	global       *rate.Limiter
	perSession   rate.Limit
	burst        int
	keyFunc      func(context.Context) string
	staleTTL     time.Duration
	reapInterval time.Duration
	reaperCancel context.CancelFunc
}

// WithRateLimit installs middleware that applies token-bucket limits to tool
// calls. By default each client session has its own bucket; calls without a
// session share a bucket. An optional global bucket can cap aggregate traffic.
// A denied call returns an IsError tool result unless OnDeny is provided.
func WithRateLimit(opts RateLimitOpts) ServerOption {
	return WithToolHandlerMiddleware(rateLimitMiddleware(opts))
}

func rateLimitMiddleware(opts RateLimitOpts) ToolHandlerMiddleware {
	store := newRateLimitStore(opts)
	return func(next ToolHandlerFunc) ToolHandlerFunc {
		return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			if !store.allow(ctx) {
				if opts.Logger != nil {
					opts.Logger.DebugContext(ctx, "tool call rate limited", "tool", request.Params.Name)
				}
				if opts.OnDeny != nil {
					return opts.OnDeny(ctx, request)
				}
				return mcp.NewToolResultError("rate limit exceeded; retry later"), nil
			}
			return next(ctx, request)
		}
	}
}

func newRateLimitStore(opts RateLimitOpts) *rateLimitStore {
	if math.IsNaN(float64(opts.PerSessionRPS)) || opts.PerSessionRPS <= 0 {
		panic("server: rate limit per-session RPS must be positive")
	}
	if opts.PerSessionBurst <= 0 {
		panic("server: rate limit per-session burst must be positive")
	}
	if math.IsNaN(float64(opts.GlobalRPS)) || opts.GlobalRPS < 0 || opts.GlobalBurst < 0 ||
		(opts.GlobalRPS == 0) != (opts.GlobalBurst == 0) {
		panic("server: rate limit global RPS and burst must both be positive or both be zero")
	}
	if opts.StaleTTL < 0 {
		panic("server: rate limit stale TTL cannot be negative")
	}

	reapInterval := opts.ReapInterval
	if reapInterval == 0 {
		reapInterval = defaultRateLimitReapInterval
	}
	staleTTL := opts.StaleTTL
	if staleTTL == 0 {
		staleTTL = defaultRateLimitStaleTTL
	}
	keyFunc := opts.KeyFunc
	if keyFunc == nil {
		keyFunc = func(ctx context.Context) string {
			session := ClientSessionFromContext(ctx)
			if session == nil {
				return ""
			}
			return session.SessionID()
		}
	}

	store := &rateLimitStore{
		entries:      make(map[string]*rateLimitEntry),
		perSession:   opts.PerSessionRPS,
		burst:        opts.PerSessionBurst,
		keyFunc:      keyFunc,
		staleTTL:     staleTTL,
		reapInterval: reapInterval,
	}
	if opts.GlobalRPS > 0 {
		store.global = rate.NewLimiter(opts.GlobalRPS, opts.GlobalBurst)
	}
	return store
}

func (s *rateLimitStore) allow(ctx context.Context) bool {
	now := time.Now()
	if s.global != nil && !s.global.AllowN(now, 1) {
		return false
	}

	key := s.keyFunc(ctx)

	s.mu.Lock()
	entry := s.entries[key]
	if entry == nil {
		entry = &rateLimitEntry{limiter: rate.NewLimiter(s.perSession, s.burst)}
		s.entries[key] = entry
	}
	entry.lastSeen = now
	entry.active++
	if s.reapInterval > 0 && s.reaperCancel == nil {
		s.startReaperLocked()
	}
	s.mu.Unlock()

	allowed := entry.limiter.AllowN(now, 1)
	s.mu.Lock()
	entry.active--
	s.mu.Unlock()
	return allowed
}

func (s *rateLimitStore) startReaperLocked() {
	ctx, cancel := context.WithCancel(context.Background())
	s.reaperCancel = cancel
	go s.reap(ctx)
}

func (s *rateLimitStore) reap(ctx context.Context) {
	ticker := time.NewTicker(s.reapInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			if s.removeStaleEntries(now) {
				return
			}
		}
	}
}

func (s *rateLimitStore) removeStaleEntries(now time.Time) bool {
	s.mu.Lock()
	for key, entry := range s.entries {
		if entry.active == 0 && now.Sub(entry.lastSeen) >= s.staleTTL {
			delete(s.entries, key)
		}
	}
	empty := len(s.entries) == 0
	cancel := s.reaperCancel
	if empty {
		s.reaperCancel = nil
	}
	s.mu.Unlock()
	if empty && cancel != nil {
		cancel()
	}
	return empty
}
