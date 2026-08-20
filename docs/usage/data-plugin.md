# Data Plugin Guide

Canonical sample 是语义协议，不要求把所有物理数据复制成 JSONL。Reader 只负责记录存储，Importer 只负责原始字段到 canonical sample 的映射。

## Built-ins

- `jsonl`
- `json_array`
- `parquet`（需要 `data` extra）
- `tar_json`
- `canonical` importer

Dataset config：

```yaml
datasets:
  - dataset_id: captions
    uri: data/captions.parquet
    importer: project-caption
    weight: 2.0
    config:
      reader: parquet
      reader_config:
        columns: [id, image, conversations]
      importer:
        image_root: data/images
```

## Plugin

```python
class MyDataPlugin:
    def register(self, readers, importers):
        readers.register("my-reader", reader_factory)
        importers.register(MyImporter())

PLUGIN = MyDataPlugin()
```

加载：

```powershell
trainomni `
  --plugin model_plugin.py:PLUGIN `
  --data-plugin data_plugin.py:PLUGIN `
  inspect data recipe.yaml --samples 4
```

Reader 必须实现：

- `__iter__`；
- `state_dict/load_state_dict`；
- immutable source fingerprint；
- `RawRecord(value, source_uri, record_index, byte_offset)`。

Importer 必须实现稳定 `importer_id/importer_version` 和 `import_record()`。Core 自动记录 source fingerprint、record index、importer version、canonical sample hash。

## Exact resume rule

Reader cursor、mixture RNG/epoch、one-sample look-ahead 和 distributed global batch group 都会进入 checkpoint。自定义 reader 如果不能表达当前位置，就不能声明 exact resume；不要用“重新 skip N 条”掩盖不可恢复的流状态。

## Security and media

Data plugin 是 Python 代码，必须由用户显式 `--data-plugin` 授权。Importer 应校验路径、checksum、MIME 和 license metadata。TAR 内 media extraction、对象存储凭据和数据库连接属于项目插件；secret 不应写入 canonical sample 或 resolved manifest。
