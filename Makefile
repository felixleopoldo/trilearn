init:
	pip install -r requirements.txt

test:
	nosetests tests

distr:
	rm -r dist build
	python setup.py bdist_wheel
	twine upload dist/*

clean:
	rm *.pyc
