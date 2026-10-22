# activity

A small in-memory activity feed with paged export.

## Feed order

Every event has a unique integer `id` and an integer `created_at` timestamp
(seconds). The feed lists events oldest first. The order is deterministic: as
long as the store is not modified, every export returns the events in the same
order, whatever the page size.

## Paging

`fetch_page(store, token, limit)` returns up to `limit` events in feed order,
plus a token for the next page. The token is `None` when the page holds fewer
than `limit` events. A full page always gets a token, and the page after it
may be empty.

- Pass `token=None` for the first page.
- Following the tokens from the first page visits every event in the store
  exactly once, in feed order, for any `limit >= 1`. This holds as long as the
  store is not modified while paging.
- Tokens are opaque strings. Clients must not build or inspect them. A
  malformed token raises `InvalidCursor`.
- `limit < 1` raises `ValueError`.

`export_all(store, page_size)` follows the tokens and returns all events in
feed order.
