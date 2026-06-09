-- QualityForge initial schema

create table if not exists runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  scenario text not null,
  summary jsonb,
  created_at timestamptz default now() not null
);

create table if not exists test_cases (
  id uuid primary key default gen_random_uuid(),
  run_id uuid references runs(id) on delete cascade not null,
  category text not null,
  title text not null,
  preconditions text,
  steps jsonb not null default '[]',
  expected_result text,
  priority text not null,
  scores jsonb,
  created_at timestamptz default now() not null
);

alter table runs enable row level security;
alter table test_cases enable row level security;

create policy "own runs"
  on runs for all
  using (auth.uid() = user_id);

create policy "own test_cases"
  on test_cases for all
  using (
    exists (
      select 1 from runs
      where runs.id = test_cases.run_id
        and runs.user_id = auth.uid()
    )
  );
