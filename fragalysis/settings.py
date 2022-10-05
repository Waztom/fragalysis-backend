"""
Django settings for fragalysis project.

Generated by 'django-admin startproject' using Django 1.11.6.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""

import os

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.excepthook import ExcepthookIntegration

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

if os.environ.get("DEBUG_FRAGALYSIS") == 'True':
    DEBUG = True

def setdummyvar(varname):
    os.environ[varname] = "1"

# CAR dummy env variables - comment out when using encrypted secrets
# This is NASTY! How to get around when running tests and NOT wanting
# to share keys in Docker image?
IBM_API_KEY=os.environ.get("IBM_API_KEY")
if not IBM_API_KEY:
    setdummyvar("IBM_API_KEY")
MANIFOLD_API_KEY=os.environ.get("MANIFOLD_API_KEY")
if not MANIFOLD_API_KEY:
    setdummyvar("MANIFOLD_API_KEY")    
MCULE_API_KEY=os.environ.get("MCULE_API_KEY")
if not MCULE_API_KEY:
    setdummyvar("MCULE_API_KEY")
SENDGRID_API_KEY=os.environ.get("SENDGRID_API_KEY")
if not SENDGRID_API_KEY:
    setdummyvar("SENDGRID_API_KEY")

# These flags are used in the upload_tset form as follows.
# Proposal Supported | Proposal Required | Proposal / View fields
# Y                  | Y                 | Shown / Required
# Y                  | N                 | Shown / Optional
# N                  | N                 | Not Shown
PROPOSAL_SUPPORTED = True
PROPOSAL_REQUIRED = True

# Authentication check when uploading files. This can be switched off for development testing if required.
# Should always be True on production.
AUTHENTICATE_UPLOAD = True

# AnonymousUser should be the first record inserted into the auth_user table.
ANONYMOUS_USER = 1

# This is set on AWX when the fragalysis-stack is rebuilt.
SENTRY_DNS = os.environ.get("FRAGALYSIS_BACKEND_SENTRY_DNS")
if SENTRY_DNS:
    # By default only call sentry in staging/production
    sentry_sdk.init(
        dsn=SENTRY_DNS,
        integrations=[DjangoIntegration(), CeleryIntegration(), RedisIntegration(), ExcepthookIntegration(always_run=True)],

        # If you wish to associate users to errors (assuming you are using
        # django.contrib.auth) you may enable sending PII data.
        send_default_pii=True
    )

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get(
    "WEB_DJANGO_SECRET_KEY", "8flmz)c9i!o&f1-moi5-p&9ak4r9=ck$3!0y1@%34p^(6i*^_9"
)

USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

ALLOWED_HOSTS = ["*"]

# DATA_UPLOAD_MAX_MEMORY_SIZE = 26214400 # 25 MB

REST_FRAMEWORK = {
    "DEFAULT_FILTER_BACKENDS": ("django_filters.rest_framework.DjangoFilterBackend",),
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.LimitOffsetPagination",
    "PAGE_SIZE": 5000,
    "DEFAULT_VERSIONING_CLASS": "rest_framework.versioning.QueryParameterVersioning",
}

# CELERY STUFF
CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_TASK_ALWAYS_EAGER = os.environ.get('CELERY_TASK_ALWAYS_EAGER', 'False').lower() in ['true', 'yes']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
# This is to stop Celery overwriting the Django logging defaults.
# CELERYD_HIJACK_ROOT_LOGGER = False

# This can be injected as an ENV var
NEOMODEL_NEO4J_BOLT_URL = os.environ.get(
    "NEO4J_BOLT_URL", "bolt://neo4j:test@neo4j:7687"
)

# Application definition
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.admin",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # My own apps
    "scoring",
    "network",
    "viewer",
    "api",
    "hypothesis",
    "hotspots",
    "media_serve",
    # The XChem database model
    "xchem_db",
    # My utility apps
    "bootstrap3",
    "guardian",
    "graphene_django",
    "django_filters",
    "mozilla_django_oidc",  # Load after auth
    "django_extensions",
    "rest_framework",
    "rest_framework.authtoken",
    "rest_framework_swagger",
    "webpack_loader",
    "django_cleanup",
    "simple_history",
    # Fragalysis evolution
    "car",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "mozilla_django_oidc.middleware.SessionRefresh",
]

AUTHENTICATION_BACKENDS = (
    "django.contrib.auth.backends.ModelBackend",
    "fragalysis.auth.KeycloakOIDCAuthenticationBackend",
    "guardian.backends.ObjectPermissionBackend",
)

STATICFILES_DIRS = [os.path.join(BASE_DIR, "fragalysis", "../viewer/static")]

STATICFILES_FINDERS = (
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
)

# mozilla_django_oidc - from documentation: https://mozilla-django-oidc.readthedocs.io/en/stable/
# Before you can configure your application, you need to set up a client with an OpenID Connect provider (OP).
# You’ll need to set up a different client for every environment you have for your site. For example,
# if your site has a -dev, -stage, and -prod environments, each of those has a different hostname and thus you
# need to set up a separate client for each one.
# you need to provide your OpenID Connect provider (OP) the callback url for your site.
# The URL path for the callback url is /oidc/callback/.
#
# Here are examples of callback urls:
#
#   http://127.0.0.1:8000/oidc/callback/ – for local development
#   https://myapp-dev.example.com/oidc/callback/ – -dev environment for myapp
#   https://myapp.herokuapps.com/oidc/callback/ – my app running on Heroku
#
# The OpenID Connect provider (OP) will then give you the following:
#
#   a client id (OIDC_RP_CLIENT_ID)
#   a client secret (OIDC_RP_CLIENT_SECRET)

# Keycloak mozilla_django_oidc - Settings
# from keyclaok (openid provider = OP) - NB these should be environment variables - not checked in
OIDC_RP_CLIENT_ID = os.environ.get("OIDC_RP_CLIENT_ID", "fragalysis-local")
OIDC_RP_CLIENT_SECRET = os.environ.get('OIDC_RP_CLIENT_SECRET')
OIDC_KEYCLOAK_REALM = os.environ.get("OIDC_KEYCLOAK_REALM",
                                     "https://keycloak.xchem-dev.diamond.ac.uk/auth/realms/xchem")

# OIDC_OP_AUTHORIZATION_ENDPOINT = "<URL of the OIDC OP authorization endpoint>"
OIDC_OP_AUTHORIZATION_ENDPOINT = os.path.join(OIDC_KEYCLOAK_REALM, "protocol/openid-connect/auth")
# OIDC_OP_TOKEN_ENDPOINT = "<URL of the OIDC OP token endpoint>"
OIDC_OP_TOKEN_ENDPOINT = os.path.join(OIDC_KEYCLOAK_REALM, "protocol/openid-connect/token")
# OIDC_OP_USER_ENDPOINT = "<URL of the OIDC OP userinfo endpoint>"
OIDC_OP_USER_ENDPOINT = os.path.join(OIDC_KEYCLOAK_REALM, "protocol/openid-connect/userinfo")
# OIDC_OP_JWKS_ENDPOINT = "<URL of the OIDC OP certs endpoint>" - This is required when using RS256.
OIDC_OP_JWKS_ENDPOINT = os.path.join(OIDC_KEYCLOAK_REALM, "protocol/openid-connect/certs")
# OIDC_OP_LOGOUT_ENDPOINT = "<URL of the OIDC OP certs endpoint>" - This is required when using RS256.
OIDC_OP_LOGOUT_ENDPOINT = os.path.join(OIDC_KEYCLOAK_REALM, "protocol/openid-connect/logout")

# Override method to also log user out from Keycloak as well as Django.
# If desired, this should be set to "fragalysis.views.keycloak_logout"
OIDC_OP_LOGOUT_URL_METHOD = os.environ.get("OIDC_OP_LOGOUT_URL_METHOD")

# LOGIN_REDIRECT_URL = "<URL path to redirect to after login>"
LOGIN_REDIRECT_URL = "/viewer/react/landing"
# LOGOUT_REDIRECT_URL = "<URL path to redirect to after logout - must be in keycloak call back if used>"
LOGOUT_REDIRECT_URL = "/viewer/react/landing"

# After much trial and error
# Using RS256 + JWKS Endpoint seems to work with no value for OIDC_RP_IDP_SIGN_KEY seems to work for authentication.
# Trying HS256 produces a "JWS token verification failed" error for some reason.
OIDC_RP_SIGN_ALGO = "RS256"
OIDC_STORE_ACCESS_TOKEN = True
OIDC_STORE_ID_TOKEN = True
# Keycloak mozilla_django_oidc - Settings - End


ROOT_URLCONF = "fragalysis.urls"

STATIC_ROOT = os.path.join(PROJECT_ROOT, "static")

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

WSGI_APPLICATION = "fragalysis.wsgi.application"

# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

CHEMCENTRAL_DB_NAME = os.environ.get("CHEMCENT_DB_NAME", "UNKOWN")

DATABASE_ROUTERS = ['xchem_db.routers.AuthRouter']

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": os.environ.get("POSTGRESQL_DATABASE", "frag"),
        "USER": os.environ.get("POSTGRESQL_USER", "fragalysis"),
        "PASSWORD": os.environ.get("POSTGRESQL_PASSWORD", "fragalysis"),
        "HOST": os.environ.get("POSTGRESQL_HOST", "database"),
        "PORT": os.environ.get("POSTGRESQL_PORT", 5432),
    }
}

if os.environ.get("BUILD_XCDB") == 'yes':
    DATABASES["xchem_db"] = {
        "ENGINE": 'django.db.backends.postgresql',
        "NAME": os.environ.get("XCHEM_NAME"),
        "USER": os.environ.get("XCHEM_USER"),
        "PASSWORD": os.environ.get("XCHEM_PASSWORD"),
        "HOST": os.environ.get("XCHEM_HOST"),
        "PORT": os.environ.get("XCHEM_PORT")
    }

if CHEMCENTRAL_DB_NAME != "UNKOWN":
    DATABASES["chemcentral"] = {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": CHEMCENTRAL_DB_NAME,
        "USER": os.environ.get("CHEMCENT_DB_USER", "postgres"),
        "PASSWORD": os.environ.get("CHEMCENT_DB_PASSWORD", "postgres"),
        "HOST": os.environ.get("CHEMCENT_DB_HOST", "postgres"),
        "PORT": 5432,
    }

# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Internationalization
# https://docs.djangoproject.com/en/1.11/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

STATIC_URL = "/static/"
MEDIA_ROOT = "/code/media/"
MEDIA_URL = "/media/"
# Swagger loging / logout
LOGIN_URL = "/accounts/login/"
LOGOUT_URL = "/accounts/logout/"

WEBPACK_LOADER = {
    "DEFAULT": {
        "BUNDLE_DIR_NAME": "bundles/",
        "STATS_FILE": os.path.join(BASE_DIR, "frontend", "webpack-stats.json"),
    }
}

GRAPHENE = {"SCHEMA": "fragalysis.schema.schema"}  # Where your Graphene schema lives

GRAPH_MODELS = {"all_applications": True, "group_models": True}

# email settings for upload key stuff
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST_USER = os.environ.get("EMAIL_USER")
# If there is an email user is defined then check the rest of the configuration is present.
# The defaults are set for the current (gamil) production configuration.
if EMAIL_HOST_USER:
    EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
    EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', True)
    EMAIL_PORT = os.environ.get('EMAIL_PORT', 587)
    EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_PASSWORD")


# DOCS_ROOT = "/code/docs/_build/html "

# Discourse settings for API calls to Discourse Platform
DISCOURSE_PARENT_CATEGORY = 'Fragalysis targets'
DISCOURSE_USER = 'fragalysis'
DISCOURSE_HOST = os.environ.get('DISCOURSE_HOST')
# Note that this can be obtained from discourse for the desired environment.
DISCOURSE_API_KEY = os.environ.get("DISCOURSE_API_KEY")

# This suffix can be set to that the different development environments posting to the same Discourse
# server can "automatically" generate different category/post titles - hopefully reducing confusion.
# It will be appended at category or post-title, e.g. "Mpro-duncan", "Mpro-staging" etc.
# Note that it is for dev systems. It is not required on production because production will have a
# dedicated Discourse server.
DISCOURSE_DEV_POST_SUFFIX = os.environ.get("DISCOURSE_DEV_POST_SUFFIX", '')

# Squonk settings for API calls to Squonk Platform.
# The environment variable SQUONK2_DMAPI_URL
# is expected by the squonk2-client package.
SQUONK2_DMAPI_URL = os.environ.get('SQUONK2_DMAPI_URL')
SQUONK2_UI_URL = os.environ.get('SQUONK2_UI_URL')

SQUONK2_MEDIA_DIRECTORY = "fragalysis-files"
SQUONK2_INSTANCE_API = "data-manager-ui/results/instance/"

# Configure django logging.
# We provide a standard formatter that emits a timestamp, the module issuing the log
# and the level name, a little like this...
#
#   2022-05-16T09:04:29 django.request ERROR # Internal Server Error: /viewer/react/landing
#
# We provide a console and rotating file handler
# (50Mi of logging in 10 files of 5M each),
# with the rotating file handler typically used for everything.
DISABLE_LOGGING_FRAMEWORK = True if os.environ.get("DISABLE_LOGGING_FRAMEWORK", "no").lower() in ["yes"] else False
LOGGING_FRAMEWORK_ROOT_LEVEL = os.environ.get("LOGGING_FRAMEWORK_ROOT_LEVEL", "INFO")
if not DISABLE_LOGGING_FRAMEWORK:
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '%(asctime)s %(name)s.%(funcName)s():%(lineno)s %(levelname)s # %(message)s',
                'datefmt': '%Y-%m-%dT%H:%M:%S'}},
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'},
            'rotating': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'maxBytes': 5_000_000,
                'backupCount': 10,
                'filename': os.path.join(BASE_DIR, 'logs/backend.log'),
                'formatter': 'simple'}},
        'loggers': {
            'asyncio': {
                'level': 'WARNING'},
            'django': {
                'level': 'WARNING'},
            'mozilla_django_oidc': {
                'level': 'WARNING'},
            'urllib3': {
                'level': 'WARNING'}},
        'root': {
            'level': LOGGING_FRAMEWORK_ROOT_LEVEL,
            'handlers': ['rotating']}}
