import time
from flask import Flask, request
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS
import json
from datetime import datetime
import asyncio

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
hashset = set()

async def long_run_task(qid):
    while qid not in hashset:
        await asyncio.sleep(0.5)
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S"), qid
    

@app.route('/send/<qid>', methods=['GET'])
async def get_query(qid):
    time, qid = await long_run_task(qid)
    return app.response_class(
        response=json.dumps({'success':'true', 'time': time, "result":qid}),
        status=200,
        mimetype='application/json',
    )
    
@app.route('/done/<qid>', methods=['GET'])
def done_task(qid):
    hashset.add(qid)
    return app.response_class(
        response=json.dumps({'success':'true'}),
        status=200,
        mimetype='application/json',
    )
    
if __name__ == "__main__":
    app.run()

    