-- ============================================================
-- DOCUMENTS
-- ============================================================

create table if not exists public.documents (
    id uuid primary key,
    user_id uuid not null
        references auth.users(id)
        on delete cascade,

    filename text not null,
    storage_path text not null,

    chunk_count integer not null default 0,

    status text not null default 'processing'
        check (status in ('processing', 'completed', 'failed')),

    uploaded_at timestamptz not null default now()
);

create index if not exists idx_documents_user_id
    on public.documents(user_id);


-- ============================================================
-- CONVERSATIONS
-- ============================================================

create table if not exists public.conversations (
    id uuid primary key default gen_random_uuid(),

    user_id uuid not null
        references auth.users(id)
        on delete cascade,

    session_id text not null,

    title text,

    created_at timestamptz not null default now(),

    unique(user_id, session_id)
);

create index if not exists idx_conversations_user_id
    on public.conversations(user_id);

create index if not exists idx_conversations_user_session
    on public.conversations(user_id, session_id);


-- ============================================================
-- MESSAGES
-- ============================================================

create table if not exists public.messages (
    id bigint generated always as identity primary key,

    conversation_id uuid not null
        references public.conversations(id)
        on delete cascade,

    user_id uuid not null
        references auth.users(id)
        on delete cascade,

    role text not null
        check (role in ('user', 'assistant')),

    content text not null,

    sources jsonb not null default '[]'::jsonb,
    metadata jsonb not null default '{}'::jsonb,

    created_at timestamptz not null default now()
);

create index if not exists idx_messages_conversation
    on public.messages(conversation_id, created_at);

create index if not exists idx_messages_user_id
    on public.messages(user_id);


-- ============================================================
-- ROW LEVEL SECURITY
-- ============================================================

alter table public.documents enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;


-- ============================================================
-- DOCUMENT POLICIES
-- ============================================================

drop policy if exists "Users can select own documents"
    on public.documents;

drop policy if exists "Users can insert own documents"
    on public.documents;

drop policy if exists "Users can update own documents"
    on public.documents;

drop policy if exists "Users can delete own documents"
    on public.documents;


create policy "Users can select own documents"
on public.documents
for select
to authenticated
using (auth.uid() = user_id);


create policy "Users can insert own documents"
on public.documents
for insert
to authenticated
with check (auth.uid() = user_id);


create policy "Users can update own documents"
on public.documents
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);


create policy "Users can delete own documents"
on public.documents
for delete
to authenticated
using (auth.uid() = user_id);


-- ============================================================
-- CONVERSATION POLICIES
-- ============================================================

drop policy if exists "Users can select own conversations"
    on public.conversations;

drop policy if exists "Users can insert own conversations"
    on public.conversations;

drop policy if exists "Users can update own conversations"
    on public.conversations;

drop policy if exists "Users can delete own conversations"
    on public.conversations;


create policy "Users can select own conversations"
on public.conversations
for select
to authenticated
using (auth.uid() = user_id);


create policy "Users can insert own conversations"
on public.conversations
for insert
to authenticated
with check (auth.uid() = user_id);


create policy "Users can update own conversations"
on public.conversations
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);


create policy "Users can delete own conversations"
on public.conversations
for delete
to authenticated
using (auth.uid() = user_id);


-- ============================================================
-- MESSAGE POLICIES
-- ============================================================

drop policy if exists "Users can select own messages"
    on public.messages;

drop policy if exists "Users can insert own messages"
    on public.messages;

drop policy if exists "Users can delete own messages"
    on public.messages;


create policy "Users can select own messages"
on public.messages
for select
to authenticated
using (auth.uid() = user_id);


create policy "Users can insert own messages"
on public.messages
for insert
to authenticated
with check (auth.uid() = user_id);


create policy "Users can delete own messages"
on public.messages
for delete
to authenticated
using (auth.uid() = user_id);


-- ============================================================
-- GRANTS
-- ============================================================

grant select, insert, update, delete
on public.documents
to authenticated;

grant select, insert, update, delete
on public.conversations
to authenticated;

grant select, insert, delete
on public.messages
to authenticated;
