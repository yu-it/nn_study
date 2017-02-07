# -*- coding: utf-8 -*-
import os
import math
import matplotlib.pyplot as plt
import pretty_midi
import treatment
import subprocess
import numpy
import data_treating.common_util as common_util

def create_testdata(min_tone, max_tone, polytone_count, number_of_tone_color, style_set, output):
    #raw data
    #normalization data
    #answer(unit_vector, index
    train_data ,train_label,eval_data ,eval_label = common_util.get_data_path("tones")
    seq_no = 0
    max_tone
    for poly in xrange(polytone_count):
        for tone_color in available_tone_colors[0: number_of_tone_color]:
            tones = [min_tone for i in xrange(poly + 1)]
            #while not((poly == 0 or min(tones) == max(tones) - 1) and tones[0] > max_tone):
            while tones[0] <= max_tone:
                vector = vectorize(tones, len(treatment.available_tones))
                if max(vector) == 1:

                    midi = create_midi([[treatment.available_tones[tone],tone_color] for tone in tones])
                    data,label = (train_data,train_label) if common_util.drawing(0.7) else (eval_data,eval_label)
                    fname_prefix = (data+ "/" + str(seq_no) + "_" + "_".join([treatment.available_tones[tone] for tone in tones]) + "_" + tone_color.replace(" ","_")).replace("#","X")
                    label_fname_prefix = (label+ "/" + str(seq_no) + "_" + "_".join([treatment.available_tones[tone] for tone in tones]) + "_" + tone_color.replace(" ","_")).replace("#","X")
                    midi.write("work.mid")
                    convert_to_wave("work.mid")
                    array, nor_array = treatment.normalization("work.wav", 1000, 0)
                    #(未)nor_arrayをcsvに
                    with open(fname_prefix + "_nor.csv","w") as w:
                        w.write(",".join([str(x) for x in nor_array]))

                    #with open(fname_prefix + "_raw.csv","w") as w:
                    #    w.write(",".join([str(x) for x in array]))

                    with open(label_fname_prefix + "_vector.txt","w") as w:
                        w.write(",".join([str(x) for x in vector]))

                    #with open(fname_prefix + "_tone.txt","w") as w:
                    #    w.write(",".join([str(x) for x in tones]))

                cursor = poly
                tones[cursor] += 1
                while tones[cursor] > max_tone and cursor > 0:
                    tones[cursor] = min_tone
                    cursor -= 1
                    tones[cursor] += 1
                seq_no += 1

    pass

def vectorize(list,indice):
    vector = numpy.zeros(indice, dtype=numpy.int32)
    for i in list:
        vector[i] += 1
    return vector


def convert_to_wave(midifile):
    subprocess.call(".\\data_treating\\tone_data\\timidity.cmd " + midifile, shell=True)
    pass

def create_midi(tones):
    pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=120)  # pretty_midiオブジェクトを作ります
    print(tones)
    try:
        for idx, (tone, tone_color) in enumerate(tones):
            program = pretty_midi.instrument_name_to_program(tone_color)
            instrument = pretty_midi.Instrument(idx)  # instrumentはトラックに相当します。
            instrument.program = program
            note_number = pretty_midi.note_name_to_number(tone) #toneは'G4'とか
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=float(0.00000001), end=float(1))  # noteはNoteOnEventとNoteOffEventに相当します。
            instrument.notes.append(note)
            pm.instruments.append(instrument)
            pass
    except:
        pass
    return pm

available_tone_colors = ["Acoustic Grand Piano",
"Bright Acoustic Piano",
"Electric Grand Piano",
"Honky-tonk Piano",
"Electric Piano 1",
"Electric Piano 2",
"Harpsichord",
"Clavi",
"Celesta",
"Glockenspiel",
"Music Box",
"Vibraphone",
"Marimba",
"Xylophone",
"Tubular Bells",
"Dulcimer",
"Drawbar Organ",
"Percussive Organ",
"Rock Organ",
"Church Organ",
"Reed Organ",
"Accordion",
"Harmonica",
"Tango Accordion",
"Acoustic Guitar (nylon)",
"Acoustic Guitar (steel)",
"Electric Guitar (jazz)",
"Electric Guitar (clean)",
"Electric Guitar (muted)",
"Overdriven Guitar",
"Distortion Guitar",
"Guitar harmonics",
"Acoustic Bass",
"Electric Bass (finger)",
"Electric Bass (pick)",
"Fretless Bass",
"Slap Bass 1",
"Slap Bass 2",
"Synth Bass 1",
"Synth Bass 2",
"Violin",
"Viola",
"Cello",
"Contrabass",
"Tremolo Strings",
"Pizzicato Strings",
"Orchestral Harp",
"Timpani",
"String Ensemble 1",
"String Ensemble 2",
"SynthStrings 1",
"SynthStrings 2",
"Choir Aahs",
"Voice Oohs",
"Synth Voice",
"Orchestra Hit",
"Trumpet",
"Trombone",
"Tuba",
"Muted Trumpet",
"French Horn",
"Brass Section",
"SynthBrass 1",
"SynthBrass 2",
"Soprano Sax",
"Alto Sax",
"Tenor Sax",
"Baritone Sax",
"Oboe",
"English Horn",
"Bassoon",
"Clarinet",
"Piccolo",
"Flute",
"Recorder",
"Pan Flute",
"Blown Bottle",
"Shakuhachi",
"Whistle",
"Ocarina",
"Lead 1 (square)",
"Lead 2 (sawtooth)",
"Lead 3 (calliope)",
"Lead 4 (chiff)",
"Lead 5 (charang)",
"Lead 6 (voice)",
"Lead 7 (fifths)",
"Lead 8 (bass + lead)",
"Pad 1 (new age)",
"Pad 2 (warm)",
"Pad 3 (polysynth)",
"Pad 4 (choir)",
"Pad 5 (bowed)",
"Pad 6 (metallic)",
"Pad 7 (halo)",
"Pad 8 (sweep)",
"FX 1 (rain)",
"FX 2 (soundtrack)",
"FX 3 (crystal)",
"FX 4 (atmosphere)",
"FX 5 (brightness)",
"FX 6 (goblins)",
"FX 7 (echoes)",
"FX 8 (sci-fi)",
"Sitar",
"Banjo",
"Shamisen",
"Koto",
"Kalimba",
"Bag pipe",
"Fiddle",
"Shanai",
"Tinkle Bell",
"Agogo",
"Steel Drums",
"Woodblock",
"Taiko Drum",
"Melodic Tom",
"Synth Drum",
"Reverse Cymbal",
"Guitar Fret Noise",
"Breath Noise",
"Seashore",
"Bird Tweet",
"Telephone Ring",
"Helicopter",
"Applause",
"Gunshot"]

