#This script sets up a FastAPI REST API on J3 so users can access time data via a browser.


from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/")
def get_time():
    return {"current_time": time.strftime("%Y-%m-%d %H:%M:%S")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



# Run this on J3 (REST API Server): python3 rest_api.py

# Check API from Any Device (Browser or Curl): curl http://192.168.1.101:8000

# Expected Output:  {"current_time":"2025-02-24 13:24:31"}


