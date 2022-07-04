# Contributing to `astronet`

Firstly, thanks for even considering making a contribution! 🤓

However, at this time, before I've submitted my PhD thesis (which hopefully will be very soon!), I
won't be accepting contributions. This is essentially to ensure all code within the repo is a solely
a reflection of my PhD work.

## Pull Requests

For code changes, please submit a pull request (PR) by:

  1. Making a relevant `issue` (please use the suitable template)
  2. Forking this repository
  3. Create a new branch labelled as: `issue/<issue-number>/<short-token-describing-work>` (for
     examples, see current branches of this repo)
  4. Ensure branch is up to date with `master`, this may require a `git rebase origin/master` before
     making the PR.
  5. Ensure all local tests pass, and the style guide is adhered to -- installing `pre-commit` will
     help with this.
  6. Make PR detailing the changes, code version and inclusion of tests.

## Tests

For any code changes, please run the test suite. See [`astronet/tests/README.md`](https://github.com/tallamjr/astronet/blob/master/astronet/tests/README.md) for more details.

## Style Guide

Our code style follows [`black`](https://github.com/psf/black), where the latest version that we use
is defined in [`.pre-commit-config.yaml`](https://github.com/tallamjr/astronet/blob/master/.pre-commit-config.yaml)

If you are using VIM, I'd suggest installing the `psf/black` vim-plugin with an indicator to the
latest version like so:

```bash
Plug 'psf/black', { 'tag': '19.10b0' }
```

Then a helpful shortcut is the have this in your `~/.vimrc`:

```bash
" ================ 'psf/black' =============
"
" Black(Python) format the visual selection
xnoremap <Leader>k :!black -q -<CR>
map <Leader>kk :Black<CR>

```
Where hitting `<Leader>kk` will format the entire file as you edit the file.
