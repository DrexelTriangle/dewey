# Dewey

Dewey is the Triangle's digitization effort of our print archives (courtesty of the Hagerty Library), via OCR. The goal of the project is to split each PDF edition into a individual articles, render them using our current website, then showcase them at archive.thetriangle.org.

## OCR

From what I've gathered, PaddleOCR is the best tool for this job, as it has very good layout detection. I've provided an example script, but you will have to play around with it of course. My advice is that once you think you have a good approach, test the script on one edition from each year. That should provide enough variety to expose issues.

## Rendering

To start with, just try and create a JSON representation of each article (including images) that you can serve to our current frontend. Then try and experiment with the formatting and maybe even different frontend frameworks to see how best you can preserve the charm of the old newspapers.

## Scraper

I've provided a Python webscraper for your convenience, the whole archive is around 28GB and a full run took me ~3.5 hours.

### Usage

Scrape the whole archive:

```bash
python3 scrape_triangle_pdfs.py --out-dir triangle_pdfs
```

Limit to a year or other collection path fragment:

```bash
python3 scrape_triangle_pdfs.py --out-dir triangle_1926 --collection-filter 1926
```

### Notes

- Existing files are skipped when their size matches Alma metadata, so interrupted
  runs can be resumed by running the same command again.
- Metadata is written as JSON Lines to `metadata.jsonl` inside the output
  directory unless `--metadata` points somewhere else.
- The default delay is `0.5` seconds between requests. Keep a nonzero delay for
  full collection runs.
