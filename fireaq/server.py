"""Tiny user-facing interface and read-only JSON API (no on-demand paid data calls)."""
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

APP_ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title='FireRisk Open (Research Prototype)', version='0.1.0',
            description='Experimental next-day VIIRS-detection probabilities, NOT official wildfire danger.')


def prediction_file():
    return Path(os.environ.get('FIRERISK_PREDICTIONS',str(APP_ROOT/'output'/'predictions.json')))

@app.get('/health')
def health():
    return {'ok':True,'predictions_exist':prediction_file().exists(),
            'research_only':True,'data_status':'no predictions until a real model has been trained and scored'}

@app.get('/api/predictions')
def predictions():
    p=prediction_file()
    if not p.exists():
        raise HTTPException(503, detail='No verified model predictions available. Train on genuine data first.')
    return json.loads(p.read_text())

@app.get('/',response_class=HTMLResponse)
def homepage():
    return (APP_ROOT/'web'/'index.html').read_text()
