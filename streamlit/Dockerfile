# python parent image

FROM python:3.10

# set working directory
WORKDIR /app

# copy current dir content
COPY . /app

# install pip dependencies
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8501

# run the aoo
CMD ["streamlit", "run", "streamlit_server.py", "--server.port=8501", "--server.address=0.0.0.0"]