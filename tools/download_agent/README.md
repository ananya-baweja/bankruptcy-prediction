# Download agent

`bpp_fetch.py` runs on a team member's own computer (standard-library Python only)
and downloads what the processing session asks for from the public IBBI, BSE and NSE
sites, into the folder it sits in. The processing session writes job files into
`jobs\`; the agent works through them in name order and logs every request under `logs\`.

* Start: double-click `START_DOWNLOADER.bat` (relaunches the agent when it is updated).
* Safety: only the hosts in `ALLOWED_HOSTS`; only writes inside its own folder; never
  opens or runs what it downloads.
* Item kinds: `save` (a file; PDFs must be complete - Content-Length and `%%EOF`),
  `collect` (the response body appended to a JSON-lines file), `split` (cut a large file
  into parts under `transfer\`, optionally gzipped first, with SHA-256 per part and for
  the whole file).
* `join_parts.py <name>.parts.json <destination>` joins the parts on the processing side,
  checks every SHA-256 and un-gzips.

Versions: 1.4 completeness check; 1.5 `split`; 1.6 `split` with `"compress": "gzip"`
(a 132 MB filing list became 6.8 MB).
