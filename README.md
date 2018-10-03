# blueprint-python-package

Blueprint for a Python package. It installs `blueprint` using `pip`.


## Usage

It can be used as a template for new Python projects. 
Just clone it:

```sh
git clone git@github.ibm.com:CHCLS/blueprint-python-package.git 
```

Change the remote to your new git repository

```sh
git remote set-url origin git@github.ibm.com:USERNAME_OR_ORGANIZATION/NEW_REPOSITORY.git
```

Now your package is installable for anyone with access to your repo

```sh
pip3 install git+ssh://git@github.ibm.com/USERNAME_OR_ORGANIZATION/NEW_REPOSITORY
```

## Suggested setup for development

Create a `virtualenv`:

```sh
python3 -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

Install the package as editable and any dependencies:

```sh
pip3 install -e .
```

## Testing and Continuous Integration

### Tests using `unitest`

The blueprint contains unit tests in the `test/` folder.
Run the tests with: 

```sh 
python3 -m unittest discover -s test -p "test_*" -v
```

The blueprint also contains a sample executable script in the `bin/` folder.
You can use this script to test other functionalities beyond unit tests.

### Continuous Integration using Travis CI

There is a working `.travis.yml` script in the root of the repository.
Once you activate the Travis CI service `https://travis.ibm.com` for your repository, 
it will take care of installing the package, running the tests and scripts using a clean VM.

You can enable notifications for build outcomes to a slack channel of your choice:

```yaml
notifications:
  slack:
    rooms:
      - ibm-research:<TOKEN>#<CHANNEL>
    on_success: always
    on_failure: always
```

See [Slack documentation](https://docs.travis-ci.com/user/notifications/#configuring-slack-notifications) for more info.
