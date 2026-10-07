# lefdRida.github.io

Personal site of Rida Lefdali, built with Jekyll and deployed to GitHub Pages by `.github/workflows/deploy.yml`.

## Run locally

```sh
bundle install
bundle exec jekyll serve
```

## Where things live

| What | Where |
| --- | --- |
| Blog posts | `_posts/` (Markdown; `$$…$$` for math) |
| Home, Blogs, Publications, CV, 404 | `_pages/` |
| Publications | `_data/publications.yml` |
| CV | `_data/resume.json` |
| Name, links, nav | `_config.yml` |
| Styles / scripts | `assets/css/main.css`, `assets/js/main.js` |

Figures in posts: `{% include figure.liquid path="assets/img/x.png" caption="…" zoomable=true %}`.
