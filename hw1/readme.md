## HW1

__Как собрать и запустить контейнер:__

1) Перейдите в директорию, содержащую папку с тестовыми данными.
2) Склонируйте репозиторий:
`git clone https://github.com/kimihailv/dla.git`
3) Соберите образ для докера: `docker build -t kim_dla -f dla/hw1/container/main.dockerfile .`
4) Запустите контейнер: `docker run -v $(pwd):/workspace --gpus="device=4" --cpuset-cpus="0-3" --memory="32gb"  --name kim_dla_hw1 -it kim_dla bash`

__Как тестировать__:

Внутри контейнера: `python -m dla.hw1.utils.test -c dla/hw1/configs/test/test_las.json -t test_data -o out.json`

__Как обучать__:

Внутри контейнера: `python -m dla.hw1.utils.pipeline -c dla/hw1/configs/las_librispeech_hdfs_ya_bpe.json`

[Ссылка на отчёт](https://bronze-colony-d62.notion.site/DLA-1-f5c6e607f9c84bb09246f6f306f7d764)
