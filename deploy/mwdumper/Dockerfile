FROM maven:alpine
RUN apk update;apk add git
WORKDIR /app
RUN git clone https://github.com/anhbh310/mwdumper
RUN cd mwdumper;ls;mvn compile;mvn package
RUN cd mwdumper/target;mv mwdumper*.jar ../../mwdumper.jar
ENTRYPOINT ["/bin/bash"]
