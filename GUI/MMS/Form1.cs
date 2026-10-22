
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
using NAudio.Wave;

namespace MusicGenreClassifier
{
    public partial class Form1 : Form
    {
        private WaveOutEvent outputDevice;
        private AudioFileReader audioFile;
        private bool isPlaying = false;

        public Form1()
        {
            InitializeComponent();
            comboBoxModel.Items.AddRange(new string[] { "DNN", "RF", "KNN", "SVM" });
            comboBoxModel.SelectedIndex = 0;
        }

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog openFileDialog = new OpenFileDialog
                {
                    Filter = "Audio Files|*.mp3;*.wav;*.ogg;*.flac"
                };

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    txtFilePath.Text = openFileDialog.FileName;
                    StopAudio();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Greška pri odabiru datoteke: {ex.Message}");
            }
        }

        private void btnPlayPause_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(txtFilePath.Text))
            {
                MessageBox.Show("Prvo odaberite audio datoteku!");
                return;
            }

            try
            {
                if (isPlaying)
                {
                    PauseAudio();
                }
                else
                {
                    PlayAudio(txtFilePath.Text);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Greška pri reprodukciji: {ex.Message}");
            }
        }

        private void PlayAudio(string filePath)
        {
            StopAudio();

            audioFile = new AudioFileReader(filePath);
            outputDevice = new WaveOutEvent();
            outputDevice.Init(audioFile);
            outputDevice.PlaybackStopped += (s, a) => StopAudio();

            outputDevice.Play();
            isPlaying = true;
            btnPlayPause.Text = "Pause";
        }

        private void PauseAudio()
        {
            outputDevice?.Pause();
            isPlaying = false;
            btnPlayPause.Text = "Play";
        }

        private void StopAudio()
        {
            outputDevice?.Stop();
            outputDevice?.Dispose();
            outputDevice = null;

            audioFile?.Dispose();
            audioFile = null;

            isPlaying = false;
            btnPlayPause.Text = "Play";
        }

        private void btnPredict_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(txtFilePath.Text))
            {
                MessageBox.Show("Prvo odaberite audio datoteku!");
                return;
            }

            string modelType = comboBoxModel.SelectedItem.ToString().ToLower(); // "dnn", "random forest", "knn"

            try
            {
                // putanje
                string pythonExe = @"C:\Users\bokib\miniconda3\envs\STROJNO\python.exe";
                string scriptPath = @"C:\Users\bokib\MMS_KV\KV6\predict_genre_TRENUTNA.py";
                string audioPath = txtFilePath.Text;

                //  Provjere postojanja datoteka
                if (!File.Exists(pythonExe))
                    throw new FileNotFoundException("Python interpreter nije pronađen!");

                if (!File.Exists(scriptPath))
                    throw new FileNotFoundException("Predikcijska skripta nije pronađena!");

                if (!File.Exists(audioPath))
                    throw new FileNotFoundException("Odabrani audio fajl ne postoji!");

                var psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"\"{scriptPath}\" \"{audioPath}\" \"{modelType}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true, 
                    CreateNoWindow = true,
                    WorkingDirectory = Application.StartupPath 
                };

                using (var process = Process.Start(psi))
                {
                    
                    string output = process.StandardOutput.ReadToEnd();
                    string error = process.StandardError.ReadToEnd();

                    process.WaitForExit();

                    // Detekcija grešaka iz skripte
                    if (!string.IsNullOrEmpty(error))
                        throw new Exception($"Python greška: {error}");

                    if (process.ExitCode != 0)
                        throw new Exception($"Skripta završila s greškom (exit code: {process.ExitCode})");

                    // Obrada rezultata
                    lblGenre.Text = $"Predviđeni žanr: {output.Trim()}";
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Greška pri predikciji: {ex.Message}\n\nDetalji:\n{ex.InnerException?.Message}");

                // Reset rezultata
                lblGenre.Text = "Predviđeni žanr: -";
            }
        }


        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            StopAudio();
            base.OnFormClosing(e);
        }

        private void buttonPredictAll_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(txtFilePath.Text))
            {
                MessageBox.Show("Prvo odaberite audio datoteku!");
                return;
            }

            try
            {
                string pythonExe = @"C:\Users\bokib\miniconda3\envs\STROJNO\python.exe";
                string scriptPath = @"C:\Users\bokib\MMS_KV\KV6\predict_all_models.py";
                string audioPath = txtFilePath.Text;

                var psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"\"{scriptPath}\" \"{audioPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Application.StartupPath
                };

                using (var process = Process.Start(psi))
                {
                    string output = process.StandardOutput.ReadToEnd();
                    string error = process.StandardError.ReadToEnd();
                    process.WaitForExit();

                    if (process.ExitCode == 0)
                    {
                        // Parsiraj rezultate
                        var lines = output.Split('\n');
                        string results = "Rezultati predikcije:\n\n";

                        foreach (var line in lines)
                        {
                            if (line.Contains(":"))
                            {
                                var parts = line.Split(':');
                                if (parts.Length == 2)
                                {
                                    results += $"{parts[0].Trim()}: {parts[1].Trim()}\n";
                                }
                            }
                        }

                        lblGenre.Text = results;
                    }
                    else
                    {
                        lblGenre.Text = "Greška pri predikciji";
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Greška: {ex.Message}");
                lblGenre.Text = "Greška pri predikciji";
            }
        }
    }
}
