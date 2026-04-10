create extension if not exists pgcrypto;

alter table public.scored_events
  add column if not exists model_version text,
  add column if not exists benchmark_id text,
  add column if not exists feature_set_version text,
  add column if not exists dataset_version text,
  add column if not exists mlflow_run_id text;

create table if not exists public.prediction_log (
  prediction_id uuid primary key default gen_random_uuid(),
  event_id text not null,
  event_datetime timestamptz not null,
  latitude float8,
  longitude float8,
  depth float8,
  magnitude float8,
  prob_7d float8,
  prob_30d float8,
  risk_7d text,
  risk_30d text,
  features_used int4,
  scored_at timestamptz not null default now(),
  model_version text not null,
  benchmark_id text not null,
  feature_set_version text not null,
  dataset_version text not null,
  mlflow_run_id text,
  created_at timestamptz not null default now()
);

create index if not exists idx_prediction_log_event_id
  on public.prediction_log(event_id);

create index if not exists idx_prediction_log_scored_at
  on public.prediction_log(scored_at desc);

create index if not exists idx_prediction_log_model_version
  on public.prediction_log(model_version);

create table if not exists public.prediction_outcomes (
  prediction_id uuid primary key references public.prediction_log(prediction_id) on delete cascade,
  event_id text not null,
  event_datetime timestamptz not null,
  latitude float8,
  longitude float8,
  magnitude float8,
  maturity_7d_at timestamptz not null,
  actual_label_7d boolean,
  evaluated_7d_at timestamptz,
  maturity_30d_at timestamptz not null,
  actual_label_30d boolean,
  evaluated_30d_at timestamptz,
  outcome_source text not null default 'usgs_fdsn_api',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_prediction_outcomes_pending_7d
  on public.prediction_outcomes(evaluated_7d_at);

create index if not exists idx_prediction_outcomes_pending_30d
  on public.prediction_outcomes(evaluated_30d_at);

create table if not exists public.model_eval_snapshots (
  snapshot_id uuid primary key default gen_random_uuid(),
  benchmark_id text not null,
  model_version text not null,
  feature_set_version text not null,
  dataset_version text not null,
  horizon text not null check (horizon in ('7d', '30d')),
  sample_size int4 not null,
  positive_rate float8,
  pr_auc float8,
  roc_auc float8,
  brier_score float8,
  window_start timestamptz,
  window_end timestamptz,
  evaluated_at timestamptz not null default now(),
  notes text
);

create index if not exists idx_model_eval_snapshots_model_version
  on public.model_eval_snapshots(model_version, evaluated_at desc);