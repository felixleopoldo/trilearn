init:
	pip install -r requirements.txt

test:
	nosetests tests

distr:
	rm -rf dist build
	python setup.py bdist_wheel
	twine upload dist/*

clean:
	rm *.pyc
