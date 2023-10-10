FROM python:3.11

# set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y python3-opencv

# copy files
COPY src/mystroke_backend /app/src/mystroke_backend
COPY pyproject.toml /app/
COPY "hand_mediapipe_85-95_(21_2)" "/app/hand_mediapipe_85-95_(21_2)"
COPY README.md /app/
COPY .python-version /app/

# install dependencies
RUN pip3 install .

# run server
CMD ["uvicorn", "src.mystroke_backend.main:app", "--host", "0.0.0.0", "--port", "5000"]

#CMD ["pip3", "list"]

# bind port
EXPOSE 5000








