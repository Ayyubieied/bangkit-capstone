#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Tunggu database postgres siap..."

    while ! nc -z $SQL_HOST $SQL_PORT; do
      sleep 0.1
    done

    echo "Database siap"
fi
echo "DATABASE VAR $DATABASE"
# perform database migration
python manage.py makemigrations
python manage.py migrate
