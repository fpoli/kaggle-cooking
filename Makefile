
run:
	@for f in $$(ls script/); do \
		echo "(*) Running $$f"; \
		python script/$$f; \
	done

linter:
	pep8 --ignore=E251 .

clean:
	@find . -name '*.pyc' -delete
	@find . -name '*.pyo' -delete
	@find . -name '__pycache__' -delete
