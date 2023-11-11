using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Data.OleDb;
using System.Data.SqlClient;


namespace LOGIN
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        OleDbConnection conn;
        string connectionString = "Provider=MSDAORA;Password=student;User ID=student"; //oracle 서버 연결


  

        private void login_Click_1(object sender, EventArgs e)
        {
            conn = new OleDbConnection(connectionString);
            try
            {
                conn.Open(); //데이터베이스 연결

                OleDbCommand cmd = new OleDbCommand();
                cmd.CommandText = "select * from member where member_id ='" + textBox1.Text + "'";
                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn;

                OleDbDataReader read = cmd.ExecuteReader(); //select 회원ID from 회원 결과
                
               
                if (!(read.Read()))
                    labelError.Text = "존재하지 않는 아이디입니다";
                else
                {
                    if (read.GetValue(2).ToString() != textBox2.Text)
                    {
                        labelError.Text = "비밀번호가 일치하지 않습니다";
                    }
                    else
                    {
                        labelError.Text = read.GetValue(1).ToString() + "님 환영합니다";
                    }
                }
                
                read.Close();
                conn.Close();
            }

            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message); //에러 메세지 
            }

            
        }

        private void loginSignupbutton_Click_1(object sender, EventArgs e)
        {
            Form2 frm = new Form2();
            frm.ShowDialog();
        }

        private void button24_Click(object sender, EventArgs e)
        {
            this.Close();
        }
    }
}



