#!/bin/sh

# perform database migration
python manage.py makemigrations
python manage.py migrate

# start server
exec "$@"
