Param(
    [switch]$Rebuild
)

$composeFile = "infra/docker-compose.yml"

if ($Rebuild) {
    docker compose -f $composeFile up -d --build
} else {
    docker compose -f $composeFile up -d
}

docker compose -f $composeFile ps
