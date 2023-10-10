FROM python:3.11.5

# set working directory
WORKDIR /app

# copy files
COPY src/mystroke_backend /app/src/mystroke_backend
COPY pyproject.toml /app/
COPY hand_efficientnet_91_224 /app/hand_efficientnet_91_224
COPY README.md /app/
COPY .python-version /app/

# install dependencies
RUN pip3 install .

# run server
CMD ["uvicorn", "src.mystroke_backend.main:app", "--host", "0.0.0.0", "--port", "5000"]

# bind port
EXPOSE 5000








