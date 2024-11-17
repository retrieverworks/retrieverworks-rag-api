#!/bin/bash

uvicorn retrieverworks.api:app --host 0.0.0.0 --port 8080 --reload --reload-exclude='public/'
