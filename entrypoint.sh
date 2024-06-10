#!/bin/sh

# wait for the database to be available
while ! nc -z db 5432; do
  sleep 0.1
done

# perform database migration
python manage.py makemigrations
python manage.py migrate

# start server
exec "$@"
