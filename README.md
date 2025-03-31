# South Dakota local bond statements

Making it easier to access bond statements filed by local governments in South Dakota.

- **Reports processed**: 2,613
- **Coverage**: 19860211 - 20250325
- **Updated**: March 31, 2025
- **Metadata**: [south-dakota-local-bond-statements.json](south-dakota-local-bond-statements.json)

**[SDCL 6-8B-19](https://sdlegislature.gov/Statutes/6-8B-19)**
> Every public body, authority, or agency issuing any general obligation, revenue, improvements, industrial revenue, special assessment, or other bonds of any type, shall, on forms provided by the secretary of state, file with the secretary of state, the following information concerning each issue of bonds:
(1)    Name of issuer;
(2)    Designation of issue;
(3)    Date of issue;
(4)    Purpose of issue;
(5)    Type of bond;
(6)    Principal amount and denomination of bond;
(7)    Paying dates of principal and interest;
(8)    Amortization schedule;
(9)    Interest rate or rates, including total aggregate interest cost.

These documents are currently stored in the South Dakota Secretary of State's old document management system, which I've complained about [elsewhere](https://github.com/cjwinchester/south-dakota-newspaper-circulation/tree/main?tab=readme-ov-file#overview). This project uses the same approach:
- Downloads the images for each page of each filing (using [`Playwright`](https://playwright.dev/python/), [`requests`](https://requests.readthedocs.io/en/latest/) and [`BeautifulSoup`](https://www.crummy.com/software/BeautifulSoup/bs4/doc/))
- Optimizes the images and assembles them into a single PDF with new metadata fields (using [`Pillow`](https://pillow.readthedocs.io/en/stable/index.html))
- Adds an OCR layer to each PDF (using [`ocrmypdf`](https://ocrmypdf.readthedocs.io/en/latest/index.html))
- Mirrors the PDF on the Internet Archive for easier/alternate retrieval (using [`internetarchive`](https://archive.org/developers/internetarchive/))
