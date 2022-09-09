FROM onceltuca/pygraphviz

RUN pip install pyrsistent==0.16.0
RUN pip install tabulate
RUN pip install trilearn==1.25

RUN apt install time 