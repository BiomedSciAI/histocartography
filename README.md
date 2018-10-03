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
it will take care of installing the package, running tests and building a docker image 
using a clean VM.

You can enable notifications for build outcomes to a slack channel of your choice:

```yaml
notifications:
  slack:
    rooms:
      - ibm-research:<TOKEN>#<CHANNEL>
    on_success: always
    on_failure: always
```

See [Travis CI documentation](https://docs.travis-ci.com/user/notifications/#configuring-slack-notifications) for more info.


## Docker support

The blueprint contains a `Dockerfile` that builds an image containing the python package.
At the moment, it is based on the image `python:3.6` and this can be adapted to your needs. 
Docker images can be stored in a docker registry for later use.
IBM TaaS offers the posibility to create an enterprise docker registry on Artifactory. 
See [here](https://pages.github.ibm.com/TAAS/tools_guide/artifactory/getting-started.html).

Deployment example:

```sh 
docker login -u "$DOCKER_USER" -p "$DOCKER_PASS"  "$DOCKER_REGISTRY"
docker build -t "${DOCKER_REGISTRY}/${DOCKER_TAG}" .
docker push "${DOCKER_REGISTRY}/${DOCKER_TAG}" 
```
