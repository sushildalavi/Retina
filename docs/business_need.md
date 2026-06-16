# Retina Business Need

Retina is framed as a full-stack visual discovery product, not a metrics demo.

## Problem

Teams often manage large image libraries with weak search tools:

- file names are inconsistent
- tags are incomplete
- users do not know exact IDs
- finding visually similar or semantically related assets takes too long

## What Retina Solves

Retina gives a user a single search surface that can:

- search by natural language
- search by example image
- combine lightweight profile signals into a content-based recommendation request

## End-to-End Flow

1. The user enters a query in the React UI.
2. FastAPI receives the request and calls the retrieval engine.
3. CLIP encodes the query and the catalog.
4. FAISS returns ranked matches from the local index.
5. The UI renders the top results immediately with captions and scores.

## Business Value

- reduces manual browsing time
- improves discovery of usable assets
- supports content, marketing, merchandising, and review workflows
- keeps the experience local and reproducible for demos and internal evaluation

## Why It Is Full Stack

- React frontend for the user workflow
- FastAPI backend for serving and request handling
- CLIP/FAISS retrieval layer for ranking
- local evaluation and report generation for product validation

