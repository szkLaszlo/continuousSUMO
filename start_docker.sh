# Run this file for the sumo container to start.
docker_list_raw=`docker ps --filter "ancestor=sumo-dev" --filter "name=docker_sumo" -q`
docker_list=($docker_list_raw)
if [ ${#docker_list[@]} -gt 1 ]; then
    if [ -z "$1" ]; then
                echo "We have several running docker containers. Please choose one and run this script again."
                docker ps --filter "ancestor=sumo-dev"
        else
                echo "We are entering the docker container number $1 with id ${docker_list[$1]}."
            docker exec -it --user=$UID ${docker_list[$1]} /bin/bash
        fi
elif [ ${#docker_list[@]} -eq 0 ]; then
        echo "We have no running docker container, so we start one."
   docker-compose -f ./docker-compose.yml run --name docker_sumo --rm --service-ports sumo
else
        echo "We are entering the only running docker container."
    docker exec -it --user=$UID ${docker_list[0]} /bin/bash
fi
