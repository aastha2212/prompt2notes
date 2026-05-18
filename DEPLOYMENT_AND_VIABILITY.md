# Deployment & Public Viability — Prompt2Notes

Points you can use when discussing how the project would work if deployed publicly and whether it is viable.

---

## 1. ChromaDB (and vector store) for multiple users

**Current state (MVP):**  
Single ChromaDB instance, one collection, one persist directory. All chunks are in one store; filtering is by `video_id` (and in practice one “session” at a time).

**If deployed publicly, two main approaches:**

### Option A: Single collection, user-scoped by metadata (recommended for start)
- **Idea:** Keep one (or a few) ChromaDB collections; add **`user_id`** to every chunk’s metadata.
- **Write:** When adding chunks, always attach `user_id` (from auth).
- **Read:** Every query uses a filter: `where={"user_id": current_user_id}` so users only ever see their own data.
- **Pros:** Simple, minimal code change, works with current ChromaDB. Good for small/medium scale.
- **Cons:** One very large collection can need tuning (e.g. indexing, sharding) later.

### Option B: Per-user or per-tenant isolation
- **Per-user collection:** One collection per user, e.g. `prompt2notes_user_{user_id}`. Same ChromaDB server, different collection names.
- **Per-user persistence path:** e.g. `./chroma_db/{user_id}/` so each user has their own ChromaDB directory.
- **Pros:** Strong isolation, easier to delete a user’s data (drop collection or delete folder).
- **Cons:** More collections/paths to manage; need to create/load the right collection per request.

**What would be implemented for public deployment:**
- Pass **`user_id`** from the auth layer into the vector store layer.
- Either: (A) store `user_id` in metadata and **filter every retrieval by `user_id`**, or (B) create/use a **per-user collection or per-user persist path**.
- Enforce that no API or UI can query without a valid `user_id` (no cross-user data leak).

---

## 2. Other major things to implement for public deployment

### Authentication & security
- **Current:** File-based auth (e.g. `users.json`), hashed passwords.
- **For production:** Move to a real database (e.g. PostgreSQL) for user accounts; use HTTPS; consider OAuth (Google/GitHub) and rate limiting per user.

### File storage (uploaded videos/images)
- **Current:** Files are temporary on the server (or in memory).
- **For production:** Persist to object storage (e.g. S3, GCP Cloud Storage) with per-user prefixes or buckets; set retention and quotas (e.g. max storage per user, max file size).

### Processing (ASR, embeddings, RAG)
- **Current:** Same server does upload, transcription, embedding, and RAG.
- **For production:** Use a **job queue** (e.g. Celery, Redis Queue, or cloud queues like SQS): user uploads → job enqueued → worker(s) run transcription and embedding; result written to vector store and linked to `user_id`. This keeps the API responsive and allows scaling workers (including GPU workers) independently.

### Caching
- **Current:** Local filesystem cache (e.g. by file hash).
- **For production:** Use a **distributed cache** (e.g. Redis) keyed by `(user_id, file_hash)` so cache is shared across app instances and survives restarts; optionally per-user quotas for cache size.

### Database (user data and metadata)
- **Current:** `users.json` for accounts.
- **For production:** **PostgreSQL** (or similar) for: users, sessions, and optionally “jobs” (e.g. video_id, user_id, status, created_at) so you can show “Your processing history” and enforce limits.

### Rate limiting and quotas
- **Implement:** Limits per user: e.g. max uploads per day, max video length, max storage. Prevents abuse and helps control cost (transcription, storage, vector DB).

### Monitoring and observability
- **Implement:** Logging (e.g. structured logs), metrics (e.g. request count, job duration, errors), and alerts. Essential for a public service.

### Backups and compliance
- **Vector store:** ChromaDB data and/or metadata backed up (e.g. periodic snapshots of persist directory or exports).
- **User data:** Backups of DB and object storage; define retention and how to handle “delete my data” (GDPR-style) by deleting from DB, object store, and vector store for that `user_id`.

---

## 3. How the project “works” when deployed publicly (summary)

- **User signs up / logs in** → identity is `user_id`.
- **User uploads a video/image** → file goes to object storage; a **job** is created (e.g. in DB) and queued.
- **Worker** picks up the job → runs ASR (and optional frame extraction), chunking, embedding; writes chunks to **ChromaDB with `user_id`** (and optionally `video_id`) in metadata; updates job status.
- **User asks a question** → API gets `user_id` from session; calls RAG with **query filtered by `user_id`** (and optionally `video_id`); returns answer.
- **User exports PDF** → same RAG result or stored summary, rendered to PDF; no access to other users’ data.
- **ChromaDB** is used as the vector store for all users, with **strict filtering by `user_id`** (and per-user or per-tenant isolation if you choose Option B).

So: **ChromaDB is shared infrastructure; user isolation is by metadata (and optionally by collection/path).**

---

## 4. Is it viable?

### Technically viable
- **Yes.** The pipeline (ASR → chunk → embed → ChromaDB → RAG) is standard and scales with:
  - Queue + workers for CPU/GPU.
  - ChromaDB (or a managed vector DB) with user-scoped filtering or per-user collections.
  - Object storage and a proper DB for users and jobs.
- **ChromaDB** can run on a single server for a lot of users; for very large scale you can move to a managed vector DB (e.g. Pinecone, Weaviate) with the same “filter by `user_id`” idea.

### Commercially / product viability
- **Use cases:** Education (lecture notes), meetings (summaries), research (video analysis), accessibility (automated notes). There is real demand.
- **Differentiators:** RAG over your own content (privacy), multi-modal (video + image), export to notes/PDF.
- **Challenges:** Cost of transcription and LLM APIs at scale; need clear pricing/quotas; competition from generic chatbots and other summarization tools.
- **Viability:** Viable as a **B2B** (e.g. universities, teams) or **freemium B2C** product if you add the deployment pieces above and control cost with limits and caching.

---

## 5. Short “talking points” for interviews or presentations

- **ChromaDB for many users:** “We’d keep a single vector store but add `user_id` to every chunk and filter every query by `user_id` so users only see their own data. For stronger isolation we could use a separate collection or persist path per user.”
- **Scaling processing:** “Heavy work (transcription, embedding) would run in background workers behind a queue so the API stays fast and we can scale workers, including GPU, independently.”
- **Storage and auth:** “Uploads would go to object storage (e.g. S3), and we’d move from file-based auth to a proper database and HTTPS.”
- **Viability:** “The stack is standard and scalable; the main work for a public launch is user-scoped data (including ChromaDB), job queue, object storage, and rate limits. The product is viable for education and enterprise if we keep costs under control with quotas and caching.”

You can paste or adapt these sections into a doc or slide deck when explaining deployment and viability.
