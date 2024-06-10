# base image  
FROM python:3.8-slim-buster

# create user and working directory
ENV APP_HOME=/home/app/web
RUN mkdir -p $APP_HOME
RUN addgroup --system app && adduser --system --group app
WORKDIR $APP_HOME

# upgrade packages
RUN apt-get update && apt-get install -y libgl1-mesa-dev
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get update && apt-get install -y --no-install-recommends netcat-traditional
RUN pip install --upgrade pip  

# copy from local work directory to docker work directory
COPY . $APP_HOME

# install project dependencies
RUN pip install -r requirements.txt 

# give access to non-root user
RUN chown -R app:app $APP_HOME
USER app

# expose port used by django 
EXPOSE 8000

# run entrypoint.sh
RUN sed -i 's/\r$//g'  $APP_HOME/entrypoint.sh
RUN chmod +x  $APP_HOME/entrypoint.sh
ENTRYPOINT ["/home/app/web/entrypoint.sh"]

# start server  
CMD ["gunicorn", "--workers=2", "project_django.wsgi", "--bind", "0.0.0.0:8000"]
