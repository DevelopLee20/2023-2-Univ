using System;
//using System.Collections.Generic;
//using System.ComponentModel;
using System.Data;
//using System.Drawing;
//using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Security.Cryptography;

namespace sha256
{
    public partial class Form1 : Form
    {
             public Form1()
        {
            InitializeComponent();
        }
        private void button1_Click(object sender, EventArgs e)
        {
            string value = txtboxINPUT.Text;
 
            // SHA256 해시 생성
            SHA256 hash = new SHA256Managed();
            byte[] bytes = hash.ComputeHash(Encoding.ASCII.GetBytes(value));

            // 16진수 형태로 문자열 결합
            StringBuilder sb = new StringBuilder();
            foreach (byte b in bytes)
                sb.AppendFormat("{0:x2}", b);

            // 문자열 출력
            txtBoxOUPUT.Text = sb.ToString();
            string sha_length = Convert.ToString(txtBoxOUPUT.Text.Length);
            //MessageBox.Show(sha_length);
        }
    }
}