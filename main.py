from fastapi import FastAPI
import uvicorn
import qna,upload
app=FastAPI()
   
app.include_router(qna.router)
app.include_router(upload.router)
@app.get("/")
async def root():
    return {"RAG APPLICATION": "LETSSS ZOOOGOOO"}




if __name__ == "__main__":
    uvicorn.run(
        host="localhost",
        port=8000,
        reload=True,
        log_level="info",
        app=app
    )
