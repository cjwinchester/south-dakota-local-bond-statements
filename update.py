from dataclasses import dataclass, field
import os
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qsl
import random
import mimetypes
from datetime import datetime, date
import logging
import subprocess
from typing import Self
from string import Formatter
from email import utils

from playwright.sync_api import sync_playwright, Page
import requests
from bs4 import BeautifulSoup
import magic
from PIL import Image, ImageEnhance
from slugify import slugify
from internetarchive import get_item
from pypdf import PdfReader, PdfWriter
from google import genai


logging.basicConfig(
    level=logging.INFO,
    format='{asctime} - {levelname} - {message}\n',
    style='{'
)

REQ_HEADERS = {
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}


def get_filesize_formatted(byte_count: int) -> str:
    """Formats byte count for humans.

    Args:
        byte_count (int): Number of bytes.

    Returns:
        byte_count_formatted (str): Byte count in a human-readable format.
    """

    byte_count_formatted  = ''

    if not byte_count:
        return byte_count_formatted

    for unit in ('', 'K', 'M', 'G', 'T'):
        if abs(byte_count) < 1024.0:
            byte_count_formatted = f'{byte_count:3.1f} {unit}B'
            return byte_count_formatted

        byte_count /= 1024.0

    return byte_count_formatted


def get_file_info(filepath: Path) -> dict:
    """Grabs basic metadata about a file.

    Args:
        filepath (Path): Local path to a file.

    Returns:
        file_info (dict): A dictionary with the keys "file_extension", "mimetype", "filesize_bytes" and "filesize_formatted".
    """

    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    mimetype, _ = mimetypes.guess_type(
        filepath,
        strict=False
    )

    mimetype_inferred = magic.from_file(
        filepath,
        mime=True
    )

    mimetype = mimetype or mimetype_inferred

    if filepath.suffix:
        file_extension = filepath.suffix
    else:
        file_extension = mimetypes.guess_extension(
            mimetype,
            strict=False
        )

    filesize_bytes = filepath.stat().st_size
    filesize_formatted = get_filesize_formatted(filesize_bytes)

    file_info = {
        'file_extension': file_extension,
        'mimetype': mimetype,
        'filesize_bytes': filesize_bytes,
        'filesize_formatted': filesize_formatted
    }

    return file_info


def get_rss_string(
    title: str,
    link: str,
    description: str,
    last_build_date: str,
    atom_link: str,
    language_code: str = 'en-us',
    items: list[str]=[]
) -> str:
    """Builds an RSS string.

        Args:
            title (str): Value for the RSS feed's `title` tag.
            
            link (str): Value for the RSS feed's `link` tag.
            
            description (str): Value for the RSS feed's `description` tag.
            
            last_build_date (str): Value for the RSS feed's `lastBuildDate` tag.
            
            atom_link (str): `href` value for the RSS feed's `atom:link` tag.
            
            language_code (str): Value for the RSS feed's `language` tag (ref: https://www.rssboard.org/rss-language-codes). Defaults to `en-us`.
            
            items (list[str]): A list of strings, each representing one RSS `item` tag.

        Returns:
            rss (str): The RSS string.
    """

    rss = f'''
    <?xml version="1.0" encoding="UTF-8" ?>
      <rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
      <channel>
        <title>{title}</title>
        <link>{link}</link>
        <description>{description}</description>
        <language>{language_code}</language>
        <lastBuildDate>{last_build_date}</lastBuildDate>
        <atom:link href="{atom_link}" rel="self" type="application/rss+xml" />
        {'\n'.join(items)}
      </channel>
      </rss>
    '''

    return rss


def build_readme(
    filepath_template: Path,
    filepath_outfile: Path = Path('README.md'),
    replacements: dict={}
) -> Path:
    """Builds a readme file.
        
        Args:
            filepath_template (Path): The template file to render.

            filepath_outfile (Path): The file to write out to. Defaults to README.md

            replacements (dict): A dictionary of replacements in which the key is the string to replace and the value the string to replace it with. Defaults to {}. Example: replacements={'{% UPDATED %}', 'May 1, 2025'}
    """

    with open(filepath_template, 'r') as infile:
        tmpl = infile.read()

    for pair in replacements.items():
        tmpl = tmpl.replace(*pair)

    with open(filepath_outfile, 'w') as outfile:
        outfile.write(tmpl)

    logging.info(f'Wrote {filepath_outfile}')

    return filepath_outfile


@dataclass
class SdSosFiling:
    """ A filing from the South Dakota Secretary of State's
        old document-management system, with methods to
        download and process separate files into a
        single, OCR'd PDF and gather some metadata along the way.

        Attributes:
            url (str): URL pointing to the filing's detail page.

            document_set_title (str): Title of the document set this filing belongs to.

            document_set_slug (str): Document set title slug.

            dir_tmp (Path): Local path to a temporary directory to store intermediate files. Defaults to Path('tmp').

            dir_pdfs (Path): Local path to a directory to store completed PDFs. Defaults to Path('pdfs').

            dir_thumbnails (Path): Local path to a directory to store thumbnails. Defaults to Path('thumbnails').

            keep_pdf (bool): Pass False if you want to delete the local copy of the PDF after it's uploaded to the Internet Archive. Defaults to True.

            thumbnail_height_px (int): Pixel height of thumbnail images. Defaults to 400.

            max_retries (int): Maximum number of retries to fetch intermediate files. Defaults to 50.

            req_headers (dict): Dictionary of headers to pass along with HTTP requests. Defaults to `REQ_HEADERS` 👆.

            gemini_prompt_system (str): System prompt for the Gemini large language model. Defaults to an empty string.

            gemini_prompt_user (str): User prompt for the Gemini large language model. Defaults to an empty string.

            gemini_api_key (str): Gemini API key. Defaults to the `GEMINI_API_KEY` environment variable or an empty string if not present.

            gemini_model (str): Name of the Gemini model to use (https://ai.google.dev/gemini-api/docs/models). Defaults to 'gemini-2.0-flash-001'.

            filing_data (dict): A dictionary of data about the filing, scraped from its row in the search results table. Defaults to {}.

            fmt_str_pdf_metadata_title (str): A format string for the PDF metadata "title" attribute. Example: '{document_set_title} - {guid}'. Defaults to an empty string.

            fmt_str_pdf_metadata_description (str): A format string for the PDF metadata "description" attribute. Example: 'A {document_type} campaign finance filing submitted on {filing_date} by {candidate} - {guid}'. Defaults to an empty string.

            ia_access (str): Internet archive API key. Defaults to the `INTERNET_ARCHIVE_ACCESS_KEY` environment variable or an empty string if not present.

            ia_secret (str): Internet archive API secret. Defaults to the `INTERNET_ARCHIVE_SECRET_KEY` environment variable or an empty string if not present.
    """

    url: str
    document_set_title: str
    document_set_slug: str
    dir_tmp: Path = Path('tmp')
    dir_pdfs: Path = Path('pdfs')
    dir_thumbnails: Path = Path('thumbnails')
    keep_pdf: bool = True
    thumbnail_height_px: int = 400
    max_retries: int = 50
    req_headers: dict = field(default_factory=lambda: REQ_HEADERS)
    gemini_prompt_system: str = ''
    gemini_prompt_user: str = ''
    gemini_api_key: str = os.environ.get(
        'GEMINI_API_KEY', ''
    )
    gemini_model: str = 'gemini-2.0-flash-001'
    filing_data: dict = field(default_factory=dict)
    fmt_str_pdf_metadata_title: str = ''
    fmt_str_pdf_metadata_description: str = ''
    ia_access: str = os.environ.get(
        'INTERNET_ARCHIVE_ACCESS_KEY', ''
    )
    ia_secret: str = os.environ.get(
        'INTERNET_ARCHIVE_SECRET_KEY', ''
    )

    def __post_init__(self):
        """Sets additional attributes and creates directories"""

        for key in self.filing_data:
            setattr(self, key, self.filing_data[key])

        self.ia_identifier = f'{self.document_set_slug}-{self.guid}'

        self.ia_pdf_link = f'https://archive.org/download/{self.ia_identifier}/{self.guid}.pdf'

        self.filing_data.update({
            'internet_archive_identifier': self.ia_identifier,
            'url_internet_archive_pdf': self.ia_pdf_link
        })

        for d in [self.dir_tmp, self.dir_pdfs, self.dir_thumbnails]:
            d.mkdir(exist_ok=True)

        self.filepath_html = self.dir_tmp / f'{self.guid}.html'
        self.filepath_pdf = self.dir_pdfs / f'{self.guid}.pdf'
        self.filepath_thumbnail = self.dir_thumbnails / f'{self.guid}.gif'

        self._filepath_gemini_data = self.dir_tmp / f'{self.guid}-extracted-data.json'

        self.filepath_images = []

        self.pub_date = utils.formatdate()

    def download_and_parse_html(self) -> Self:
        """Download the filing's detail page and parse
           the HTML into a BeautifulSoup object.

            Returns:
                Self
        """

        if self.filepath_html.exists():
            with open(self.filepath_html, 'r') as infile:
                self.html = infile.read()
                self.soup = BeautifulSoup(self.html, 'html.parser')
            return self

        retries = self.max_retries
        url = self.url
        guid = self.guid
        req_headers = self.req_headers

        def attempt_download(retries=retries):
            """A recursive function that tries to download the HTML page.

            Args:
                retries (int): Number of attempts to download the file before throwing an exception.

            Raises:
                AssertionError, Exception
            """

            r = requests.get(
                url,
                headers=req_headers
            )

            r.raise_for_status()

            time.sleep(random.uniform(2, 4))

            try:
                assert 'document thumbnails' in r.text.lower()

                with open(self.filepath_html, 'w') as outfile:
                    outfile.write(r.text)

                logging.info(f'Wrote {self.filepath_html}')

                self.html = r.text
                self.soup = BeautifulSoup(r.text, 'html.parser')
                return self
            except AssertionError:
                time.sleep(random.uniform(2, 4))

                retries -= 1

                if retries == 0:
                    raise Exception(f'Could not retrieve page for {guid}')

                logging.warning(f'Failed to download {url}. Trying again ... ({retries})')

                attempt_download(
                    retries=retries
                )

        attempt_download()

        return self

    def download_and_process_images(self) -> Self:
        """Downloads the filing's separate images, boosts their
           sharpness and contrast and creates a thumbnail.

           Raises:
               Exception

           Returns:
               Self
        """

        # skip if PDF already exists
        if self.filepath_pdf.exists():
            return self

        if not self.html or not self.soup:
            self.download_and_parse_html()

        retries = self.max_retries
        guid = self.guid
        req_headers = self.req_headers
        detail_page_url = self.url

        img_links = self.soup.find('table').find_all('a')
        img_links = [urljoin(self.url, x.get('href')) for x in img_links]

        logging.info(f'Attempting to download images ({len(img_links):,})')

        # most frequent case is no preassembled PDF
        self.filepath_preassembled_pdf = None

        def attempt_download(retries=retries):

            for url in img_links:

                qs = {x[0].split('?')[-1]: x[1] for x in parse_qsl(url)}

                image_number = qs.get('Image', '')

                filepath_stub = f'{guid}-{str(image_number).zfill(3)}'

                already_downloaded = list(
                    self.dir_tmp.glob(f'{filepath_stub}*')
                )

                if already_downloaded:
                    if already_downloaded[0] not in self.filepath_images and 'pdf' not in str(already_downloaded[0]):
                        self.filepath_images.append(already_downloaded[0])
                    continue

                s = requests.Session()

                s.get(
                    detail_page_url,
                    headers=req_headers
                )

                time.sleep(random.uniform(2, 4))

                r = s.get(
                    url,
                    headers=req_headers
                )

                r.raise_for_status()

                time.sleep(random.uniform(2, 4))

                mimetype = r.headers.get('content-type').split(';')[0].strip().lower()

                file_extension = mimetypes.guess_extension(
                    mimetype,
                    strict=False
                ) or ''

                # failed requests return error page
                if mimetype == 'text/html':

                    logging.info(f'Failed attempt to download {url}\nRetrying ... {retries}')

                    retries -= 1

                    if retries == 0:
                        raise Exception(f'Could not download image at {url}')
                    
                    attempt_download(retries=retries)

                    continue

                filepath = self.dir_tmp / f'{filepath_stub}{file_extension}'

                with open(filepath, 'wb') as outfile:
                    outfile.write(r.content)

                logging.info(f'Downloaded {filepath}')

                # some records included preassembled PDFs in addition
                # to images of the pages
                if file_extension == '.pdf':
                    self.filepath_preassembled_pdf = filepath
                
                if filepath not in self.filepath_images and file_extension != '.pdf':
                    self.filepath_images.append(filepath)

            return self.filepath_images

        attempt_download()

        self.filepath_images.sort()

        self.images = []

        for img_path in self.filepath_images:
            with Image.open(img_path) as im:
                ImageEnhance.Contrast(im).enhance(50.0)
                ImageEnhance.Sharpness(im).enhance(50.0)

                if im not in self.images:
                    self.images.append(im)

        # delete HTML file
        self.filepath_html.unlink()
        logging.info(f'Deleted {self.filepath_html}')       

        if self.filepath_thumbnail.exists():
            return self

        if self.filepath_images:

            # create a thumbnail
            size = self.thumbnail_height_px, self.thumbnail_height_px

            with Image.open(self.filepath_images[0]) as im:
                im.thumbnail(size)
                im.save(
                    self.filepath_thumbnail
                )

            logging.info(f'Made a thumbnail: {self.filepath_thumbnail}')

        return self

    def build_pdf(self) -> Self:
        """Checks to see if the "image" set includes a PDF, else
        conditionally assembles the processed images into a PDF file. Calls `ocrmypdf` to OCR the PDF and collect some file metadata.

           Raises:
               AssertionError

           Returns:
               Self
        """

        # check to see if the completed PDF already exists
        if self.filepath_pdf.exists():
            reader = PdfReader(
                str(self.filepath_pdf)
            )

            self.pages = len(reader.pages)
            self.filing_data['pages'] = self.pages

            file_info = get_file_info(self.filepath_pdf)

            self.filesize_bytes = file_info['filesize_bytes']
            self.filesize_formatted = file_info['filesize_formatted']

            self.filing_data['filesize_bytes'] = self.filesize_bytes
            self.filing_data['filesize_formatted'] = self.filesize_formatted

            # delete images
            for image_path in self.filepath_images:
                image_path.unlink()
                logging.info(f'Deleted {image_path}')

            reader.close()

            return self

        if not getattr(self, 'images'):
            self.download_and_process_images()

        img_count = len(self.images)

        self.pages = img_count
        self.filing_data['pages'] = self.pages

        # build metadata strings

        # https://stackoverflow.com/a/22830468
        # first assert that any variables in the PDF metadata format
        # strings are also attributes of this object
        fmt_variables_title = [fn for _, fn, _, _ in Formatter().parse(self.fmt_str_pdf_metadata_title) if fn is not None]

        fmt_variables_description = [fn for _, fn, _, _ in Formatter().parse(self.fmt_str_pdf_metadata_title) if fn is not None]

        assert not set(fmt_variables_description + fmt_variables_title).difference(set(self.__dict__.keys()))

        # build the title string
        fmt_dict_title = {x: getattr(self, x) for x in fmt_variables_title}
        pdf_title = self.fmt_str_pdf_metadata_title.format(**fmt_dict_title)

        # build the description string
        fmt_dict_description = {x: getattr(self, x) for x in fmt_variables_description}
        pdf_description = self.fmt_str_pdf_metadata_description.format(**fmt_dict_description)

        metadata = {}

        if pdf_title:
            self.pdf_title = pdf_title
            metadata['title'] = pdf_title

        if pdf_description:
            self.pdf_description = pdf_description
            metadata['subject'] = pdf_description

        self.images[0].save(
            self.filepath_pdf,
            'PDF',
            resolution=180.0,
            save_all=True,
            append_images=self.images[1:],
            **metadata
        )

        logging.info(f'Wrote {self.filepath_pdf}')

        # check to see if there's a preassembled PDF
        if self.filepath_preassembled_pdf:
            reader = PdfReader(
                str(self.filepath_preassembled_pdf)
            )

            page_count = len(reader.pages)

            # overwrite with preassembled PDF only if it has more pages than there are images
            if page_count > img_count:

                self.pages = page_count
                self.filing_data['pages'] = self.pages

                writer = PdfWriter()

                # add pages to the writer
                for page in reader.pages:
                    writer.add_page(page)

                md_pairs = {f'/{k.title()}': v for k, v in metadata}

                # add the metadata
                writer.add_metadata(md_pairs)

                # save the PDF to file
                with open(self.filepath_pdf, 'wb') as f:
                    writer.write(f)

                writer.close()

            reader.close()

            # delete the preassembled PDF
            time.sleep(1)
            self.filepath_preassembled_pdf.unlink()
            logging.info(f'Deleted {self.filepath_preassembled_pdf}')

        cmds = [
            'ocrmypdf',
            '--deskew',
            '--rotate-pages',
            '--clean',
            self.filepath_pdf,
            self.filepath_pdf,
        ]

        subprocess.run(cmds)

        file_info = get_file_info(self.filepath_pdf)
        self.filesize_bytes = file_info['filesize_bytes']
        self.filesize_formatted = file_info['filesize_formatted']

        self.filing_data['filesize_bytes'] = self.filesize_bytes
        self.filing_data['filesize_formatted'] = self.filesize_formatted

        # delete images
        for image_path in self.filepath_images:
            image_path.unlink()
            logging.info(f'Deleted {image_path}')

        return self

    @property
    def rss_item(self) -> Self:
        """An RSS item with data about this filing."""

        self.rss_item = f'''
        <item>
          <title>New filing - {self.pdf_title}</title>
          <link>{self.ia_pdf_link}</link>
          <description>{self.pdf_description} (source: {self.source_url})</description>
          <pubDate>{self.pub_date}</pubDate>
          <guid isPermaLink="false">{self.guid}</guid>
        </item>
        '''

        return self

    def upload_pdf_to_ia(self) -> Self:
        """Upload PDF to the Internet Archive"""

        item = get_item(self.ia_identifier)

        metadata = self.filing_data.copy()

        r = item.upload(
            files=str(self.filepath_pdf),
            access_key=self.ia_access,
            secret_key=self.ia_secret,
            metadata=metadata,
            verbose=True,
            retries=50,
            retries_sleep=2
        )
        
        logging.info(f'Uploaded PDF to the Internet Archive: {self.ia_pdf_link}')

        if not self.keep_pdf:
            # delete PDF file
            self.filepath_pdf.unlink()
            logging.info(f'Deleted {self.filepath_pdf}')

        return self

    def _gemini_extract_data(self) -> Self:
        """Uses the Gemini large language model to extract
           data from the PDF based on the prompts, then writes
           results to file.

           Returns:
               Self
        """
        if not self.pdf_filepath.exists():
            self.build_pdf()

        if self._gemini_data_filepath.exists():
            return self

        client = genai.Client(
            api_key=self.gemini_api_key
        )

        response = client.models.generate_content(
            model=self.gemini_model,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=self.gemini_prompt_system
            ),
            contents=[
                types.Part.from_bytes(
                    data=self.pdf_filepath.read_bytes(),
                    mime_type='application/pdf',
                ),
                self.gemini_prompt_user
            ]
        )

        self.extracted_data = response.text

        with open(self._gemini_data_filepath, 'w') as outfile:
            outfile.write(self.extracted_data)

        logging.info(f'Wrote {self._gemini_data_filepath}')

        return self

    def scrape(self, ia_upload: bool=False) -> Self:
        """Main function to scrape this filing."""

        self.download_and_parse_html()
        self.download_and_process_images()
        self.build_pdf()

        if ia_upload:
            self.upload_pdf_to_ia()

        return self

    def __str__(self):
        return f'S.D. Secretary of State document - {self.document_set_title} - {self.guid}'


@dataclass
class SdSosDocumentSet:
    """An object to manage the extraction of documents
       and metadata from a document set stored in the
       South Dakota Secretary of State's old document-management system.

    Args:
        search_url (str): The URL of the document set's search page.

        title (str): The name of this document set.
        table_headers (list): A list of header values to use as column names in the search results.

        dir_pdfs (Path): Local path to a directory where completed PDFs will be stored. Defaults to Path('pdfs').

        dir_filings (Path): Local path to a directory to store intermediate files. Defaults to Path('filings').

        dir_data_file (Path): Local path to the directory where the metadata file will be created. Defaults to Path('.').

        max_retries (int): Number of times to retry loading the search results if an exception is thrown. Defaults to 10.

        post_processing_funcs (dict): A dictionary with zero or more table_header strings keyed to functions to process values of that type. Defaults to {}.

        fmt_str_pdf_metadata_title (str): A format string for the PDF metadata "title" attribute into which you can later interpolate variables. Example: '{document_set_title} - {guid}'. Defaults to an empty string.

        fmt_str_pdf_metadata_description (str): A format string for the PDF metadata "description" attribute into which you can later interpolate variables. Example: 'A {document_type} campaign finance filing submitted on {filing_date} by {candidate} - {guid}'. Defaults to an empty string.

        keep_pdfs (bool): Pass False if you want to delete the local copies of the PDFs after they're uploaded to the Internet Archive. Defaults to True.

        sort_columns (tuple): Columns to sort the output JSON array, in order. Defaults to ().

        filename_data (str): Custom name for the data file that's written out. Defaults to {self.slug}.json.

        filename_rss (str): Custom name for the RSS file that's written out. Defaults to {self.slug}.rss.

        skip_guids (list): A list of guids to skip.
    """

    search_url: str
    title: str
    slug: str
    table_headers: list[str]
    dir_pdfs: Path = Path('pdfs')
    dir_data_file: Path = Path('.')
    max_retries: int = 10
    req_headers: dict = field(default_factory=lambda: REQ_HEADERS)
    post_processing_funcs: dict = field(default_factory=dict)
    fmt_str_pdf_metadata_title: str = ''
    fmt_str_pdf_metadata_description: str = ''
    keep_pdfs: bool = True
    sort_columns: tuple = ()
    filename_data: str = ''
    filename_rss: str = ''
    skip_guids: list = field(default_factory=list)

    def __post_init__(self):
        """Gets a list of completed guids, creates the slug,
           creates directories if needed, sets the data filepath,
           ensures post-processing keys are present in the
           table headers.

           Raises:
               Exception
        """

        for colname in self.post_processing_funcs:
            if colname not in self.table_headers:
                raise Exception(f'{colname} not in {self.table_headers}')

        self.dir_pdfs.mkdir(exist_ok=True)
        self.dir_data_file.mkdir(exist_ok=True)

        self.filepath_data = self.dir_data_file / self.filename_data or f'{self.slug}.json'
        self.filepath_rss = self.dir_data_file / self.filename_rss or f'{self.slug}.xml'

        current_data, completed_guids = [], []

        if self.filepath_data.exists():
            with open(self.filepath_data, 'r') as infile:
                data = json.load(infile)
                current_data = data
                completed_guids = [x['guid'] for x in data]

        self.current_data = current_data
        self.completed_guids = completed_guids

        # get/set a list of identifiers for filings that
        # already were uploaded to the Internet archive
        self.get_current_ia_identifiers()

    def get_search_results(self) -> Self:
        """Launch a Playwright browser to load the search
           results and parse the table HTML.

           Sets:
               self.search_results: A list of dictionaries with some basic metadata on each filing returned in the search results.

           Raises:
               AssertionError

           Returns:
               Self
        """
        try:
            with sync_playwright() as p:
                browser = p.firefox.launch(headless=False)
                context = browser.new_context()

                page = context.new_page()

                page.goto(
                    self.search_url,
                    timeout=0
                )

                time.sleep(random.uniform(5, 7))

                page.locator(f'input#cmdSubmit').click()

                table = page.locator('span#lbl_SearchInfo > table')

                time.sleep(random.uniform(5, 7))

                self.soup = BeautifulSoup(
                    table.inner_html(),
                    'html.parser'
                )

                browser.close()

            rows = self.soup.find_all('tr')
            self.search_results = []

            # loop over the table rows, skipping the headers
            for row in rows[1:]:
                cells = row.find_all('td')

                # throw if the header count doesn't match the
                # td count
                assert len(self.table_headers) == len(cells)

                row_data = dict(zip(
                    self.table_headers,
                    [x.text.strip() for x in cells]
                ))

                # find the link to the detail page
                url_partial = row.find('a').get('href')

                # grab fully qualified detail page URL
                row_data['source_url'] = urljoin(
                    self.search_url,
                    url_partial
                )

                # grab the doc guid
                row_data['guid'] = {x[0].split('?')[-1]: x[1] for x in parse_qsl(row_data['source_url'])}['DocGuid']

                # apply any post-processing functions to the data
                for colname in self.post_processing_funcs:
                    func = self.post_processing_funcs[colname]
                    row_data[colname] = func(row_data[colname])

                self.search_results.append(row_data)
        except:
            time.sleep(random.uniform(4, 6))
            self.get_search_results()

        logging.info(f'Found {len(self.search_results):,} search results.')

        return self

    def fetch_documents(self) -> Self:
        """Loops over the table of search results
           creates an SdSosFiling object for each filing, then
           scrapes the filing data and writes to file.

            Returns:
                Self
        """
        if not self.search_results:
            self.get_search_results()

        self.new_filings = []

        for filing in self.search_results:
            guid = filing.get('guid')

            if guid in self.skip_guids:
                continue

            url = filing.get('source_url')

            filing_object = SdSosFiling(
                url,
                self.title,
                self.slug,
                keep_pdf=self.keep_pdfs,
                filing_data=filing,
                fmt_str_pdf_metadata_title=self.fmt_str_pdf_metadata_title,
                fmt_str_pdf_metadata_description=self.fmt_str_pdf_metadata_description,
                max_retries=self.max_retries
            )

            # is this filing already on the Internet Archive?
            ia_file_uploaded_already = filing_object.ia_identifier in self.current_ia_identifiers

            if guid in self.completed_guids and ia_file_uploaded_already:
                continue

            filing_object.scrape(
                ia_upload=not ia_file_uploaded_already
            )

            # add filing data to running list
            self.current_data.append(
                filing_object.filing_data
            )

            # dump to file to save state
            with open(self.filepath_data, 'w') as outfile:
                json.dump(self.current_data, outfile)

            self.new_filings.append(filing_object)

        with open(self.filepath_data, 'w') as outfile:
            json.dump(
                self.current_data,
                outfile,
                indent=4
            )

        logging.info(f'Wrote {self.filepath_data}')

        return self

    def get_current_ia_identifiers(self) -> Self:
        """Fetch a list of identifiers for filings already
           uploaded to the Internet archive.
        """

        url = 'https://archive.org/advancedsearch.php'

        params = {
            'q': self.slug,
            'fl[]': 'identifier',
            'rows': len(self.current_data)+1,
            'output': 'json'
        }

        r = requests.get(
            url,
            headers=self.req_headers,
            params=params
        )

        r.raise_for_status()

        if not r.ok:
            time.sleep(random.uniform(2, 4))
            logging.warning('Failed to reach the internet archive - trying again ...')
            self.get_current_ia_identifiers()

        self.current_ia_identifiers = [x['identifier'] for x in r.json()['response']['docs']]

        return self

    @property
    def rss_string(self) -> str:
        """An RSS string with items from the self.new_filings list. """

        return get_rss_string(
            title=self.title,
            link=self.search_url,
            description=f'An RSS feed flagging new filings - {self.title}',
            last_build_date=utils.formatdate(),
            atom_link=self.atom_link,
            language_code=self.language_code,
            items=[x.rss_item for x in self.new_filings]
        )

    def write_rss_file(self) -> Self:
        """Writes an RSS file built from the `self.new_filings` list (if any) to `self.filepath_rss`."""

        if not self.new_filings:
            return Self

        with open(self.filepath_rss, 'w') as outfile:
            outfile.write(self.rss_string)

        logging.info(f'Wrote {self.filepath_rss}')

        return Self

    def __str__(self):
        return f'South Dakota Secretary of State document set: {self.title}'


if __name__ == '__main__':

    # i have a request into the SOS office to get copies
    # of these reports
    skip_guids = [
        # stalls trying to download images 2 and 5
        '325c7386-ed6d-43d4-b40d-7e365190623b',

        # stalls trying to download the page
        '3c2bff40-dfd0-45bb-a78d-565bee574435',

        # stalls trying to download image 46
        '3c91727d-ec8f-4860-b96e-62c5ffb90aca',

        # stalls trying to download the page
        '69cdd41a-bded-4415-995b-160c5ab1d505',

        # stalls trying to download the page
        'a7ccd110-acac-451d-b1d7-87be4dacf4d9',

        # stalls downloading the page
        'b2fdca2c-eecb-41d9-8766-f793246984e9',

        # misplaced athletic agent form
        '20210614-1528-5970-7457-20b12a80f05a',

        # misplaced athletic agent form
        '20210614-1528-4172-5118-3a27e918fe16'
    ]

    search_url = 'https://sdsos.gov/general-information/miscellaneous-forms/local-bond-statements/search/'

    document_set_title ='South Dakota local bond statements'
    document_set_slug = slugify(document_set_title)

    project_url = f'https://github.com/cjwinchester/{document_set_slug}'

    atom_feed = project_url.replace('github.com', 'raw.githubusercontent') + 'refs/heads/main/{document_set_slug}.xml'

    document_set = SdSosDocumentSet(
        search_url,
        document_set_title,
        document_set_slug,
        table_headers=[
            'issuer',
            'date_issued',
            'date_filed'
        ],
        fmt_str_pdf_metadata_title='{document_set_title} - {issuer} - {date_issued} - {guid}',
        fmt_str_pdf_metadata_description='A bond statement filed by {issuer} on {date_issued}.',
        filename_data='south-dakota-local-bond-statements.json',
        max_retries=200,
        skip_guids=skip_guids
    )

    document_set.get_search_results()
    document_set.fetch_documents()

    # sort the data and write to file
    document_set.current_data.sort(
        key=lambda x: (
            x['date_issued'],
            x['date_filed'],
            x['issuer']
        )
    )

    with open(document_set.filepath_data, 'w') as outfile:
        json.dump(
            document_set.current_data,
            outfile,
            indent=4
        )

    logging.info(f'Wrote {document_set.filepath_data}')

    rss_items = [x.rss_item for x in document_set.new_filings if x.rss_item]

    if rss_items:
        write_rss_feed(
            title=document_set_title,
            link=project_url,
            description='An RSS feed that flags new bond statements filed by local governments in South Dakota.',
            last_build_date=utils.formatdate(),
            atom_link=atom_feed,
            items=rss_items
        )

    dates = [x['date_issued'] for x in document_set.current_data if x['date_issued']]

    report_total = len(document_set.current_data)

    build_readme(
        'readme.template',
        'README.md',
        {
            '{% TOTAL_REPORTS %}': f'{report_total:,}',
            '{% EARLIEST_DATE %}': str(min(dates)),
            '{% LATEST_DATE %}': str(max(dates)),
            '{% UPDATED_DATE %}': datetime.now().strftime('%B %d, %Y')
        }
    )
