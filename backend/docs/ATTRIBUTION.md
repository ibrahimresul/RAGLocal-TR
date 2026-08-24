# Data Sources

The document corpus in this directory was compiled from Turkish Wikipedia.
Wikipedia content is available under the
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
This Markdown file is not indexed; the application processes only TXT, PDF,
and DOCX files.

| File | Source |
|---|---|
| 01-software.docx | https://tr.wikipedia.org/wiki/Yazılım |
| 02-software-engineering.docx | https://tr.wikipedia.org/wiki/Yazılım_mühendisliği |
| 03-object-oriented-programming.docx | https://tr.wikipedia.org/wiki/Nesne_yönelimli_programlama |
| 04-database-management-system.docx | https://tr.wikipedia.org/wiki/Veritabanı_yönetim_sistemi |
| 05-cloud-computing.docx | https://tr.wikipedia.org/wiki/Bulut_bilişim |
| 06-cybersecurity.docx | https://tr.wikipedia.org/wiki/Bilgisayar_güvenliği |
| 07-artificial-intelligence.docx | https://tr.wikipedia.org/wiki/Yapay_zeka |
| 08-git-version-control.docx | https://tr.wikipedia.org/wiki/Git_(yazılım) |
| 09-data-structures.docx | https://tr.wikipedia.org/wiki/Veri_yapısı |
| 10-operating-systems.docx | https://tr.wikipedia.org/wiki/İşletim_sistemi |

## Turkish Stopwords

`app/stopwords.py` contains the Turkish list from the official
[NLTK stopwords data package](https://github.com/nltk/nltk_data/raw/gh-pages/packages/corpora/stopwords.zip).
The downloaded archive had SHA-256
`48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b`.
The NLTK corpus README identifies PostgreSQL Snowball stopwords as the upstream
source.
