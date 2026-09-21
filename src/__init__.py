"""
src package
===========
Core modules for the Face Recognition Identification System.

Sub-modules
-----------
detector    – SCRFD face detection
embedder    – ArcFace embedding generation
matcher     – Cosine-similarity matching with unknown rejection
enrollment  – Enrollment pipeline (detect → embed → store)
recognition – Identification pipeline (detect → embed → match)
utils       – Shared helpers (image I/O, drawing, formatting)
"""
