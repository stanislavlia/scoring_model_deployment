# 
FROM tiangolo/uvicorn-gunicorn:python3.10

# 

# 
COPY ./requirements.txt /app/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r requirements.txt


# 
COPY ../app  /app/app

RUN ls -l
# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
