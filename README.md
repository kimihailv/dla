# Homeworks of DLA Course
## HW1

Как тестировать:

1) Перейдите в директорию, содержащую папку с тестовыми данными.
2) Склонируйте репозиторий:
`git clone https://github.com/kimihailv/dla.git`
3) Соберите образ для докера: `docker build -t kim_dla -f dla/hw1/container/main.dockerfile .`
4) Запустите контейнер: `docker run -v $(pwd):/workspace --gpus="device=4" --cpuset-cpus="0-3" --memory="32gb"  --name kim_dla_hw1 -it kim_dla bash`
5) Внутри контейнера: `python -m dla.hw1.utils.test -c dla/hw1/configs/test/test_las.json -t test_data -o out.json`