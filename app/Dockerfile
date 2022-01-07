FROM python:3.9.9
EXPOSE 8501
WORKDIR /app
COPY requirement.txt requirement.txt
RUN pip3 install --no-cache-dir -r requirement.txt
RUN apt update -y && apt upgrade -y && apt install libsndfile1 -y && apt install ffmpeg -y
COPY . .
ENTRYPOINT streamlit run app.py