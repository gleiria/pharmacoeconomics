
# need to check what this is (she is using a nodejs app as an example)
# do I need to add my python environment?
FROM python:3.9.6

# sets working directory inside the cointainer. 
# sets the base working directory inside the container
WORKDIR /web_tool

# install application dependencies
COPY requirements.txt ./

# this is the local codebase
COPY web_tool ./ 

# RUN is an image build step
# install application dependencies -> for example pip install

RUN pip install -r requirements.txt

# This is the command that will run when the container starts up on the host
# I can ignore this and run on remote only
#CMD [ "executable" ]

# do not get this one very well but is related to ports etc.. 
#EXPOSE port