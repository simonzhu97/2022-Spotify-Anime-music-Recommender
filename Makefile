all: image run

image: app/templates/*.html app.py requirements.txt
	docker build -f dockerfiles/Dockerfile.app -t flask_app .

run: config/local/config app.py
	docker run --env-file config/local/config --rm -p 5000:5000 flask_app

clean:
	docker image rm -f flask_app