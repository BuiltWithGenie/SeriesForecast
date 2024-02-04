# pull latest julia image
FROM julia:latest
# create dedicated user
RUN useradd --create-home --shell /bin/bash genie
# set up the app
RUN mkdir /home/genie/app
COPY . /home/genie/app
WORKDIR /home/genie/app

# configure permissions
RUN chown -R genie:genie /home/
# switch user
USER genie
# instantiate Julia packages
RUN julia -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate(); Pkg.precompile();"

# ports
EXPOSE 8000
EXPOSE 80
# set up app environment
ENV JULIA_DEPOT_PATH "/home/genie/.julia"
ENV JULIA_REVISE = "off"
ENV GENIE_ENV "prod"
ENV GENIE_HOST "0.0.0.0"
ENV PORT "8000"
ENV WSPORT "8000"
# run app
ENTRYPOINT ["julia", "--project", "-e", "using GenieFramework; Genie.loadapp(); up(async=false);"]
