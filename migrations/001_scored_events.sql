-- migrations/001_scored_events.sql
-- Run once in Supabase SQL editor (or via psql)
-- Table: scored_events — stores all earthquake scores produced by the API

CREATE TABLE IF NOT EXISTS public.scored_events (
    id              bigint          GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Event identity (USGS id or null for manual inputs)
    event_id        text            UNIQUE,

    -- Location & source
    latitude        double precision NOT NULL,
    longitude       double precision NOT NULL,
    depth           double precision,               -- km
    magnitude       double precision NOT NULL,
    event_time      timestamptz     NOT NULL,        -- UTC time of the earthquake

    -- Model outputs
    prob_7d         double precision,               -- P(M≥5 within 7d, 200km)
    prob_30d        double precision,               -- P(M≥5 within 30d, 200km)
    prob_365d       double precision,               -- P(M≥5 within 365d, 200km)
    risk_7d         text,                           -- human-readable risk label
    risk_30d        text,
    risk_365d       text,
    features_used   integer,                        -- number of non-NaN features

    -- Metadata
    scored_at       timestamptz     NOT NULL DEFAULT now()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_scored_events_scored_at
    ON public.scored_events (scored_at DESC);

CREATE INDEX IF NOT EXISTS idx_scored_events_prob_7d
    ON public.scored_events (prob_7d DESC)
    WHERE prob_7d IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_scored_events_location
    ON public.scored_events USING gist (
        point(longitude, latitude)
    );

-- Row-level security: allow public read (for Lovable/Grafana), no public write
ALTER TABLE public.scored_events ENABLE ROW LEVEL SECURITY;

-- Anyone can read (Lovable frontend, Grafana)
CREATE POLICY "Allow public read"
    ON public.scored_events
    FOR SELECT
    USING (true);

-- Only service_role can insert/update (the Python API)
-- (service_role bypasses RLS by default — no extra policy needed)

COMMENT ON TABLE public.scored_events IS
    'Earthquake sequence forecasting scores — one row per scored event.';
COMMENT ON COLUMN public.scored_events.prob_7d IS
    'Probability of at least one M≥5.0 follow-up within 200km over 7 days.';
COMMENT ON COLUMN public.scored_events.prob_30d IS
    'Probability of at least one M≥5.0 follow-up within 200km over 30 days.';
COMMENT ON COLUMN public.scored_events.prob_365d IS
    'Probability of at least one M≥5.0 follow-up within 200km over 365 days.';
