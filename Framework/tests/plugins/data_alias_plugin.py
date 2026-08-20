from __future__ import annotations

from trainomni.data import CanonicalImporter


class AliasImporter(CanonicalImporter):
    importer_id = "canonical-alias"


class DataAliasPlugin:
    def register(self, readers, importers):
        importers.register(AliasImporter())


PLUGIN = DataAliasPlugin()
