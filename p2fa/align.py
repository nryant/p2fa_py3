#!/usr/bin/env python3

""" Command-line usage:
      python3 align.py [options] wave_file transcript_file output_dir
      where options may include:
        -r sampling_rate -- override which sample rate model to use,
                           one of 8000, 11025, and 16000 (default: 16000)
        -s start_time    -- start of portion of wavfile to align
                           (in seconds, default 0)
        -e end_time      -- end of portion of wavfile to align
                           (in seconds, default to end)
        -t state_align   -- align HMM states (eg. s1, s2, s3)
                           additionally; default=0
        -v verbose       -- print HCopy and HVite commandline; default=0

    output_dir is created if it does not exist. Three output files are
    written to that directory, each sharing the base name of the input
    wav file:
        <basename>.TextGrid  -- Praat TextGrid with phone and word
                               interval tiers
        <basename>.words     -- word-level alignment in HTK label format
        <basename>.phones    -- phone-level alignment in HTK label format

    HTK label files are tab-delimited with one segment per line:
        onset (s) <TAB> offset (s) <TAB> label

    You can also import this file as a module and use the functions directly.

    2018-02-22 JK, This file was modified for Python3.x
    2018-08-21  papagandalf, This file was modified so that it can be
                called from Python code
    2020-06-20 JK, command-line option fixed;
                   verbose option added for debugging;
                   state-level alignment added;
    2026-03-11  default sampling rate changed to 16000;
                output_dir replaces output_file at command line;
                .words and .phones HTK label files now written alongside
                TextGrid;
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import wave


LOG_LIKELIHOOD_REGEX = r'.+==\s+\[\d+ frames\]\s+(-?\d+.\d+)'

# HTK stores timestamps in units of 100 ns; divide by this to get seconds.
HTK_TIME_UNIT = 10000000.0

# Half the HTK PLP analysis window duration (25 ms window => 12.5 ms).
# Added to raw HTK timestamps so that onsets/offsets correspond to the
# centres of the analysis windows rather than their beginnings.
HTK_TS_SHIFT = 0.0125

# HTK internally treats 11025 Hz audio as if it were sampled at 11000 Hz,
# so its timestamps are slightly stretched relative to real time.
# Multiplying by this factor scales them back to true wall-clock time.
HTK_SR_11025_SCALE = 11000.0 / 11025.0


def htk_to_seconds(htk_time):
    """Convert an HTK timestamp from 100 ns units to seconds.

    Parameters
    ----------
    htk_time : int or float
        Raw HTK timestamp in 100 ns units.

    Returns
    -------
    float
        Time in seconds.
    """
    return float(htk_time) / HTK_TIME_UNIT


def prep_wav(orig_wav, out_wav, sr_override, wave_start, wave_end, sr_models):
    """Prepare a wav file for HTK alignment, resampling or trimming as needed.

    Reads the sample rate of *orig_wav*. If the rate is not in *sr_models*,
    differs from *sr_override*, or a time range has been requested, the file
    is resampled and/or trimmed with sox and written to *out_wav*. Otherwise
    *orig_wav* is copied to *out_wav* unchanged.

    Parameters
    ----------
    orig_wav : str
        Path to the original input wav file.
    out_wav : str
        Path to write the prepared wav file.
    sr_override : int or None
        If not None, force resampling to this sample rate (Hz).
    wave_start : str
        Start time of the region to align, in seconds.
    wave_end : str or None
        End time of the region to align, in seconds. If None, align to
        the end of the file.
    sr_models : list of int or None
        Sample rates for which acoustic models are available. If the
        file's native rate is not in this list it will be resampled.
        Pass None to skip this check.

    Returns
    -------
    int
        Sample rate (Hz) of the prepared wav file.
    """
    if os.path.exists(out_wav) and False:
        f = wave.open(out_wav, 'r')
        sr = f.getframerate()
        f.close()
        print("Already re-sampled the wav file to " + str(sr))
        return sr

    with wave.open(orig_wav, 'r') as f:
        sr = f.getframerate()

    soxopts = []
    if float(wave_start) != 0.0 or wave_end is not None:
        soxopts += ['trim', wave_start]
        if wave_end is not None:
            soxopts += [str(float(wave_end) - float(wave_start))]

    if (sr_models is not None and sr not in sr_models) or (
            sr_override is not None and sr != sr_override) or soxopts:
        new_sr = 11025
        if sr_override is not None:
            new_sr = sr_override

        print("Resampling wav file from " + str(sr) +
              " to " + str(new_sr) + "...")
        sr = new_sr
        subprocess.run(
            ['sox', orig_wav, '-r', str(sr), out_wav] + soxopts,
            check=True)
    else:
        shutil.copy(orig_wav, out_wav)

    return sr


def prep_mlf(trsfile, mlffile, word_dictionary, surround, between):
    """Prepare an HTK Master Label File (MLF) from a plain-text transcript.

    Reads *trsfile*, normalises the text (upper-casing, stripping
    punctuation, expanding common non-speech tokens), filters out any
    words not found in *word_dictionary*, and writes the result to
    *mlffile* in HTK MLF format.

    Parameters
    ----------
    trsfile : str
        Path to the plain-text transcript file.
    mlffile : str
        Path to write the output MLF file.
    word_dictionary : str
        Path to the HTK pronunciation dictionary. Only words present in
        this dictionary will be included; others are skipped with a
        warning.
    surround : str or None
        Comma-separated token(s) to insert at the beginning and end of
        the word sequence (e.g. ``'sp'``). Pass None to omit.
    between : str or None
        Token to insert between every pair of consecutive words
        (e.g. ``'sp'`` for an optional silence). Pass None to omit.
    """
    # Read in the dictionary to ensure all of the words
    # we put in the MLF file are in the dictionary. Words
    # that are not are skipped with a warning.
    with open(word_dictionary, 'r') as f:
        the_dict = {}  # build hash table
        for line in f.readlines():
            if line != "\n" and line != "":
                the_dict[line.split()[0]] = True

    with open(trsfile, 'r') as f:
        lines = f.readlines()

    words = []

    if surround is not None:
        words += surround.split(',')

    i = 0

    # this pattern matches hyphenated words, such as TWENTY-TWO;
    # however, it doesn't work with longer things like SOMETHING-OR-OTHER
    hyphen_pat = re.compile(r'([A-Z]+)-([A-Z]+)')

    while i < len(lines):
        txt = lines[i].replace('\n', '')
        txt = txt.replace('{breath}', '{BR}').replace('&lt;noise&gt;', '{NS}')
        txt = txt.replace('{laugh}', '{LG}').replace('{laughter}', '{LG}')
        txt = txt.replace('{cough}', '{CG}').replace('{lipsmack}', '{LS}')

        for pun in [',', '.', ':', ';', '!', '?', '"', '%',
                    '(', ')', '--', '---']:
            txt = txt.replace(pun, '')

        txt = txt.upper()

        # break up any hyphenated words into two separate words
        txt = re.sub(hyphen_pat, r'\1 \2', txt)

        txt = txt.split()

        for wrd in txt:
            if wrd in the_dict:
                words.append(wrd)
                if between is not None:
                    words.append(between)
            else:
                print("SKIPPING WORD", wrd)

        i += 1

    # remove the last 'between' token from the end
    if between is not None:
        words.pop()

    if surround is not None:
        words += surround.split(',')

    write_input_mlf(mlffile, words)


def write_input_mlf(mlffile, words):
    """Write a word-level HTK Master Label File (MLF) for alignment input.

    Parameters
    ----------
    mlffile : str
        Path to the output MLF file.
    words : list of str
        Ordered list of word labels to write into the MLF.
    """
    with open(mlffile, 'w') as fw:
        fw.write('#!MLF!#\n')
        fw.write('"*/tmp.lab"\n')
        for wrd in words:
            fw.write(wrd + '\n')
        fw.write('.\n')


def read_aligned_mlf(mlffile, sr, wave_start, duration=None):
    """Read an HTK MLF alignment output file and return phone/word timings.

    Parses the phone-level alignment produced by HVite and returns a
    nested list grouping phones under their parent words. Timestamps are
    converted from raw HTK 100 ns units to seconds, shifted by half the
    PLP analysis window duration (``HTK_TS_SHIFT``) so that they
    correspond to window centres, and offset by *wave_start*. For 11025
    Hz audio an additional scaling correction is applied
    (``HTK_SR_11025_SCALE``) because HTK internally treats that rate as
    11000 Hz.

    The first phone onset is clamped to *wave_start* and the last phone
    offset is clamped to ``wave_start + duration`` to counteract any
    overshoot introduced by the timestamp shift.

    Parameters
    ----------
    mlffile : str
        Path to the HTK MLF alignment output file.
    sr : int
        Sample rate (Hz) of the aligned audio.
    wave_start : float or str
        Start time (seconds) of the aligned region within the original
        recording. Added to all timestamps so that they are expressed in
        the coordinate system of the original file.
    duration : float or None
        Duration (seconds) of the aligned audio segment. Used to clamp
        the final offset. If None, the duration is estimated from the
        last parsed phone offset before clamping.

    Returns
    -------
    list of list
        One sub-list per word. Each sub-list begins with the word label
        (str) followed by zero or more phone entries of the form
        ``[label, onset, offset]`` where *onset* and *offset* are
        times in seconds.

    Raises
    ------
    ValueError
        If the MLF file contains fewer than 3 lines, indicating that
        alignment did not complete successfully.
    """
    # TODO: extract log-likelihood score
    with open(mlffile, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    if len(lines) < 3:
        raise ValueError("Alignment did not complete succesfully.")

    j = 2
    ret = []
    while lines[j] != '.':
        # Is this the start of a word; do we have a word label?
        if len(lines[j].split()) >= 5:
            # Make a new word list in ret and put the word label
            # at the beginning
            wrd = lines[j].split()[4]
            ret.append([wrd])

        # Append this phone to the latest word (sub-)list
        ph = lines[j].split()[2]
        if sr == 11025:
            st = (htk_to_seconds(lines[j].split()[0])
                  + HTK_TS_SHIFT) * HTK_SR_11025_SCALE
            en = (htk_to_seconds(lines[j].split()[1])
                  + HTK_TS_SHIFT) * HTK_SR_11025_SCALE
        else:
            st = htk_to_seconds(lines[j].split()[0]) + HTK_TS_SHIFT
            en = htk_to_seconds(lines[j].split()[1]) + HTK_TS_SHIFT
        if st < en:
            ret[-1].append([ph, st + wave_start, en + wave_start])

        j += 1

    # If no duration was supplied, estimate it from the last parsed offset
    # (before clamping), using the same conversion pipeline as above.
    if duration is None:
        duration = ret[-1][-1][2] - float(wave_start)

    # Clamp first onset and last offset to eliminate the effect of the
    # HTK_TS_SHIFT applied to all timestamps above.
    ret[0][1][1] = float(wave_start)
    ret[-1][-1][2] = float(wave_start) + duration

    return ret


def make_alignment_lists(word_alignments):
    """Flatten word-grouped alignments into separate phone and word lists.

    Parameters
    ----------
    word_alignments : list of list
        Output of :func:`read_aligned_mlf`. Each sub-list begins with a
        word label followed by phone entries of the form
        ``[label, onset, offset]``.

    Returns
    -------
    phons : list of list
        Flat list of all phone entries ``[label, onset, offset]`` in
        order.
    wrds : list of list
        One entry per realised word (words with no phones, such as
        optional silences that were not realised, are omitted). Each
        entry is ``[label, onset, offset]`` where *onset* is the start
        of the first phone and *offset* is the end of the last phone.
    """
    # make the list of just phone alignments
    phons = []
    for wrd in word_alignments:
        phons.extend(wrd[1:])  # skip the word label

    # make the list of just word alignments
    # we're getting elements of the form:
    #   ["word label", ["phone1", start, end], ["phone2", start, end], ...]
    wrds = []
    for wrd in word_alignments:
        # If no phones make up this word, then it was an optional word
        # like a pause that wasn't actually realized.
        if len(wrd) == 1:
            continue
        # word label, first phone start time, last phone end time
        wrds.append([wrd[0], wrd[1][1], wrd[-1][2]])
    return phons, wrds


def get_av_log_likelihood_per_frame(file_path):
    """Parse the average log-likelihood per frame from an HVite results file.

    Parameters
    ----------
    file_path : str
        Path to the HVite results file (``aligned.results``).

    Returns
    -------
    float
        Average log-likelihood per frame reported by HVite.
    """
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    score = re.match(LOG_LIKELIHOOD_REGEX, lines[-1]).groups()[0]

    return float(score)


def write_text_grid(outfile, word_alignments, state_alignments=None):
    """Write a Praat TextGrid file from phone and word alignments.

    Creates a TextGrid in short text format with an ``IntervalTier`` for
    phones and an ``IntervalTier`` for words. If *state_alignments* are
    provided, a third ``IntervalTier`` for HMM states is prepended.

    Parameters
    ----------
    outfile : str
        Path to write the output TextGrid file.
    word_alignments : list of list
        Output of :func:`read_aligned_mlf`. Each sub-list begins with a
        word label followed by phone entries of the form
        ``[label, onset, offset]``.
    state_alignments : list of list or None
        Optional HMM state alignments in the same nested format as
        *word_alignments*, with phone labels as the outer grouping.
        If None, no state tier is written.
    """
    # make the list of just phone alignments
    phons = []
    for wrd in word_alignments:
        phons.extend(wrd[1:])  # skip the word label

    # make the list of just state alignments
    if state_alignments is not None:
        states = []
        for sts in state_alignments:
            states.extend(sts[1:])  # skip the phone label

    # make the list of just word alignments
    # we're getting elements of the form:
    #   ["word label", ["phone1", start, end], ["phone2", start, end], ...]
    wrds = []
    for wrd in word_alignments:
        # If no phones make up this word, then it was an optional word
        # like a pause that wasn't actually realized.
        if len(wrd) == 1:
            continue
        # word label, first phone start time, last phone end time
        wrds.append([wrd[0], wrd[1][1], wrd[-1][2]])

    with open(outfile, 'w') as fw:
        fw.write('File type = "ooTextFile short"\n')
        fw.write('"TextGrid"\n')
        fw.write('\n')
        fw.write(str(phons[0][1]) + '\n')
        fw.write(str(phons[-1][-1]) + '\n')
        fw.write('<exists>\n')
        if state_alignments is not None:
            fw.write('3\n')
        else:
            fw.write('2\n')

        # write the state interval tier
        if state_alignments is not None:
            fw.write('"IntervalTier"\n')
            fw.write('"state"\n')
            fw.write(str(states[0][1]) + '\n')
            fw.write(str(states[-1][-1]) + '\n')
            fw.write(str(len(states)) + '\n')
            for k in range(len(states)):
                fw.write(str(states[k][1]) + '\n')
                fw.write(str(states[k][2]) + '\n')
                fw.write('"' + states[k][0] + '"' + '\n')

        # write the phone interval tier
        fw.write('"IntervalTier"\n')
        fw.write('"phone"\n')
        fw.write(str(phons[0][1]) + '\n')
        fw.write(str(phons[-1][-1]) + '\n')
        fw.write(str(len(phons)) + '\n')
        for k in range(len(phons)):
            fw.write(str(phons[k][1]) + '\n')
            fw.write(str(phons[k][2]) + '\n')
            fw.write('"' + phons[k][0] + '"' + '\n')

        # write the word interval tier
        fw.write('"IntervalTier"\n')
        fw.write('"word"\n')
        fw.write(str(phons[0][1]) + '\n')
        fw.write(str(phons[-1][-1]) + '\n')
        fw.write(str(len(wrds)) + '\n')
        for k in range(len(wrds) - 1):
            fw.write(str(wrds[k][1]) + '\n')
            fw.write(str(wrds[k+1][1]) + '\n')
            fw.write('"' + wrds[k][0] + '"' + '\n')
        fw.write(str(wrds[-1][1]) + '\n')
        fw.write(str(phons[-1][2]) + '\n')
        fw.write('"' + wrds[-1][0] + '"' + '\n')


def write_htk_label_file(outfile, segments):
    """Write an HTK label file with one segment per line.

    Parameters
    ----------
    outfile : str
        Path to the output HTK label file.
    segments : list of list
        Each element is ``[label, onset, offset]`` where *onset* and
        *offset* are times in seconds.
    """
    with open(outfile, 'w') as f:
        for label, onset, offset in segments:
            f.write(f'{onset}\t{offset}\t{label}\n')


def prep_scp(wavfile, tmpdir):
    """Write the HTK SCP script files required by HCopy and HVite.

    Creates two files in *tmpdir*:

    * ``codetr.scp`` — maps *wavfile* to the PLP feature file path,
      used by HCopy to generate features.
    * ``test.scp`` — lists the PLP feature file path, used by HVite
      during Viterbi decoding.

    Parameters
    ----------
    wavfile : str
        Path to the prepared wav file to be feature-extracted.
    tmpdir : str
        Path to the temporary working directory for this run.
    """
    with open(os.path.join(tmpdir, 'codetr.scp'), 'w') as fw:
        fw.write(wavfile + ' ' + os.path.join(tmpdir, 'tmp.plp') + '\n')
    with open(os.path.join(tmpdir, 'test.scp'), 'w') as fw:
        fw.write(os.path.join(tmpdir, 'tmp.plp') + '\n')


def create_plp(hcopy_config, tmpdir, verbose=False):
    """Run HCopy to extract PLP features from the prepared wav file.

    Reads the wav file listed in ``codetr.scp`` and writes PLP features
    to ``tmp.plp`` in *tmpdir*, using the supplied HCopy configuration
    file.

    Parameters
    ----------
    hcopy_config : str
        Path to the HCopy configuration file for the target sample rate.
    tmpdir : str
        Path to the temporary working directory for this run.
    verbose : bool, optional
        If True, print the HCopy command before executing it.
        Default is False.
    """
    cmd = [
        'HCopy', '-T', '1',
        '-C', hcopy_config,
        '-S', os.path.join(tmpdir, 'codetr.scp')]
    if verbose:
        print('creating plp...\n', ' '.join(cmd))

    subprocess.run(cmd, check=True)


def viterbi(input_mlf, word_dictionary, output_mlf, phoneset, hmmdir,
            tmpdir, state_align=False, verbose=False):
    """Run HVite to perform Viterbi forced alignment.

    Aligns the PLP features in ``tmp.plp`` in *tmpdir* against the word
    sequence in *input_mlf* and writes phone-level (and optionally
    HMM state-level) alignments to *output_mlf*. HVite stdout (which
    contains the per-utterance log-likelihood score) is written to
    ``aligned.results`` in *tmpdir*.

    Parameters
    ----------
    input_mlf : str
        Path to the input MLF file containing the word sequence.
    word_dictionary : str
        Path to the HTK pronunciation dictionary.
    output_mlf : str
        Path to write the output phone-level alignment MLF.
    phoneset : str
        Path to the HTK phoneset file (``monophones`` or ``hmmnames``).
    hmmdir : str
        Directory containing the acoustic model files (``macros`` and
        ``hmmdefs``).
    tmpdir : str
        Path to the temporary working directory for this run.
    state_align : bool, optional
        If True, pass ``-f -y lab`` to HVite to produce HMM state-level
        alignments in addition to phone-level ones. Default is False.
    verbose : bool, optional
        If True, print the HVite command before executing it.
        Default is False.
    """
    cmd = ['HVite', '-T', '1', '-a', '-m']
    if state_align:
        cmd += ['-f', '-y', 'lab']
    cmd += [
        '-I', input_mlf,
        '-H', os.path.join(hmmdir, 'macros'),
        '-H', os.path.join(hmmdir, 'hmmdefs'),
        '-S', os.path.join(tmpdir, 'test.scp'),
        '-i', output_mlf,
        '-p', '0.0',
        '-s', '5.0',
        word_dictionary,
        phoneset]

    if verbose:
        print('running viterbi...\n', ' '.join(cmd))

    results_file = os.path.join(tmpdir, 'aligned.results')
    with open(results_file, 'w') as f:
        subprocess.run(cmd, check=True, stdout=f)


def align(wavfile, trsfile, outdir=None, wave_start='0.0', wave_end=None,
          sr_override=None, model_path=None, custom_dict=None,
          state_align=False, verbose=False):
    """Forced-align a wav file to its transcript using P2FA acoustic models.

    Prepares input files, runs HCopy (PLP feature extraction) and HVite
    (Viterbi decoding), and returns phone- and word-level alignments. If
    *outdir* is provided the alignments are also written to disk as a
    Praat TextGrid and a pair of HTK label files.

    Parameters
    ----------
    wavfile : str
        Path to the input wav file.
    trsfile : str
        Path to the plain-text transcript file.
    outdir : str or None, optional
        Directory in which to write output files. Created if it does not
        exist. Three files are written, each named after *wavfile*:
        ``<stem>.TextGrid``, ``<stem>.words``, and ``<stem>.phones``.
        If None, no files are written. Default is None.
    wave_start : str, optional
        Start time (seconds) of the portion of *wavfile* to align.
        Default is ``'0.0'``.
    wave_end : str or None, optional
        End time (seconds) of the portion of *wavfile* to align.
        If None, align to the end of the file. Default is None.
    sr_override : int or None, optional
        Force a specific sample rate (Hz) for alignment. Must be one of
        the rates for which an acoustic model exists (8000, 11025, or
        16000) when using the built-in models. Default is None.
    model_path : str or None, optional
        Path to the directory containing acoustic model files. If None,
        the ``model`` subdirectory next to this script is used and
        sample-rate-specific subdirectories are selected automatically.
        Default is None.
    custom_dict : str or None, optional
        Path to an additional pronunciation dictionary to append to the
        built-in one. If None, a ``dict.local`` file in the current
        working directory is used if present. Default is None.
    state_align : bool or int, optional
        If True (or 1), also produce HMM state-level alignments.
        Default is False.
    verbose : bool or int, optional
        If True (or 1), print HCopy and HVite commands before executing
        them. Default is False.

    Returns
    -------
    phoneme_alignments : list of list
        Flat list of phone-level segments, each ``[label, onset, offset]``
        with times in seconds.
    word_alignments : list of list
        List of word-level segments, each ``[label, onset, offset]`` with
        times in seconds. Words with no realised phones are omitted.
    state_alignments : list of list
        HMM state-level segments in the same format as *phoneme_alignments*.
        Only returned when *state_align* is True.
    av_score_per_frame : float
        Average Viterbi log-likelihood per frame.
    """
    surround_token = "sp"
    between_token = "sp"

    # If no model directory was said explicitly, get directory
    # containing this script.
    hmmsubdir = ""
    sr_models = None
    if model_path is None:
        model_path = os.path.dirname(os.path.realpath(__file__)) + "/model"
        hmmsubdir = "FROM-SR"
        # sample rates for which there are acoustic models set up, otherwise
        # the signal must be resampled to one of these rates.
        sr_models = [8000, 11025, 16000]

    if (sr_override is not None and sr_models is not None
            and sr_override not in sr_models):
        raise Exception("invalid sample rate: not an acoustic model available")

    tmpdir = tempfile.mkdtemp(prefix='p2fa_')
    try:
        word_dictionary = os.path.join(tmpdir, 'dict')
        input_mlf = os.path.join(tmpdir, 'tmp.mlf')
        output_mlf = os.path.join(tmpdir, 'aligned.mlf')
        results_mlf = os.path.join(tmpdir, 'aligned.results')
        if state_align:
            state_mlf = os.path.join(tmpdir, 'aligned_state.mlf')
        else:
            state_mlf = None

        # create ./tmp/dict by concatenating our dict with a local one
        dict_files = [os.path.join(model_path, 'dict')]
        if custom_dict is not None:
            dict_files.append(custom_dict)
        elif os.path.exists('dict.local'):
            dict_files.append('dict.local')
        with open(word_dictionary, 'w') as out:
            for path in dict_files:
                with open(path, 'r') as f:
                    out.write(f.read())

        # prepare wavefile: do a resampling if necessary
        tmpwav = os.path.join(tmpdir, 'sound.wav')
        sr = prep_wav(wavfile, tmpwav, sr_override, wave_start, wave_end,
                      sr_models)

        if hmmsubdir == "FROM-SR":
            hmmsubdir = str(sr)

        # prepare mlfile
        prep_mlf(trsfile, input_mlf, word_dictionary,
                 surround_token, between_token)

        # prepare scp files
        prep_scp(tmpwav, tmpdir)

        # generate the plp file using a given configuration file for HCopy
        create_plp(
            os.path.join(model_path, hmmsubdir, 'config'),
            tmpdir,
            verbose=verbose)

        # run Viterbi decoding
        mpfile = os.path.join(model_path, 'monophones')
        if not os.path.exists(mpfile):
            mpfile = os.path.join(model_path, 'hmmnames')

        hmmdir = os.path.join(model_path, hmmsubdir)
        viterbi(input_mlf, word_dictionary, output_mlf, mpfile, hmmdir,
                tmpdir, verbose=verbose)

        # compute actual recording duration for timestamp clamping
        with wave.open(tmpwav, 'r') as f:
            rec_duration = f.getnframes() / sr

        if state_align:
            viterbi(input_mlf, word_dictionary, state_mlf, mpfile, hmmdir,
                    tmpdir, state_align=True, verbose=verbose)
            state_alignments = read_aligned_mlf(
                state_mlf, sr, float(wave_start), duration=rec_duration)
        else:
            state_alignments = None

        _alignments = read_aligned_mlf(
            output_mlf, sr, float(wave_start), duration=rec_duration)
        phoneme_alignments, word_alignments = make_alignment_lists(_alignments)

        av_score_per_frame = get_av_log_likelihood_per_frame(results_mlf)

        # output the alignment as a Praat TextGrid, plus .words/.phones
        # HTK label files
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(wavfile))[0]
            write_text_grid(
                os.path.join(outdir, stem + '.TextGrid'),
                _alignments,
                state_alignments=state_alignments)
            write_htk_label_file(
                os.path.join(outdir, stem + '.words'),
                word_alignments)
            write_htk_label_file(
                os.path.join(outdir, stem + '.phones'),
                phoneme_alignments)

    finally:
        shutil.rmtree(tmpdir)

    if not state_align:
        return phoneme_alignments, word_alignments, av_score_per_frame
    else:
        return (phoneme_alignments, word_alignments,
                state_alignments, av_score_per_frame)


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(
        description='P2FA for Python3'
                    ' (https://github.com/jaekookang/p2fa_py3)',
        add_help=True)
    parser.add_argument('wavfile', metavar='WAVFILE', type=str,
                        help='Provide wav file with valid path')
    parser.add_argument('trsfile', metavar='TRSFILE', type=str,
                        help='Provide transcription file (txt)'
                             ' with valid path')
    parser.add_argument('outdir', metavar='OUTDIR', type=str,
                        help='Output directory for alignment files'
                             ' (.TextGrid, .words, .phones)')
    parser.add_argument('-r', '--sampling_rate', metavar='SR', type=int,
                        default=16000, choices=[8000, 11025, 16000],
                        help='override which sample rate model to use,'
                             ' one of 8000, 11025, and 16000')
    parser.add_argument('-s', '--start_time', metavar='START', default='0.0',
                        help='start of portion of wavfile to align'
                             ' (in seconds, default 0)')
    parser.add_argument('-e', '--end_time', metavar='END', default=None,
                        help='end of portion of wavfile to align'
                             ' (in seconds, default to end)')
    parser.add_argument('-t', '--state_align', metavar='STATE_ALIGN', type=int,
                        default=0, choices=[0, 1],
                        help='align HMM states (eg. s1, s2, s3)'
                             ' additionally; default=0')
    parser.add_argument('-v', '--verbose', metavar='VERBOSE', type=int,
                        default=0, choices=[0, 1],
                        help='print HCopy and HVite commandlines; default=0')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    align(args.wavfile, args.trsfile, outdir=args.outdir,
          wave_start=args.start_time, wave_end=args.end_time,
          sr_override=args.sampling_rate, model_path=None, custom_dict=None,
          state_align=int(args.state_align),
          verbose=int(args.verbose))
