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

namespace LOGIN
{
    public partial class Form2 : Form
    {
        public Form2()
        {
            InitializeComponent();
        }
        bool idcheck = false;
        OleDbConnection conn;
        string connectionString = "Provider=MSDAORA;Password=student;User ID=student"; //oracle 서버 연결

        private void button1_Click(object sender, EventArgs e)
        {
            conn = new OleDbConnection(connectionString);
            try
            {
                if (txtID.Text.Length < 3)
                {
                    MessageBox.Show("ID는 3자 이상이어야 합니다");
                    idcheck = false;
                    return;
                }

                conn.Open(); //데이터베이스 연결
                OleDbCommand cmd = new OleDbCommand();
                cmd.CommandText = "select * from member where member_id ='" + txtID.Text + "'";
                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn;

                OleDbDataReader read = cmd.ExecuteReader(); //select 회원ID from 회원 결과


                if (!(read.Read()))
                { 
                    idcheck = true;
                    MessageBox.Show("사용가능한 ID입니다"); //에러 메세지 
                }
                else
                {
                    MessageBox.Show("중복 ID입니다"); //에러 메세지 
                }

                read.Close();
                conn.Close();
            }

            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message); //에러 메세지 
            }

        }

        private void button5_Click(object sender, EventArgs e)
        {
                if (txtPW.Text.Length < 2)
                {
                    MessageBox.Show("Pass Word는 3자 이상이어야 합니다");
                    return;

                }
            if (!idcheck)
            {
                MessageBox.Show("ID 중복확인을 해주세요");
                return;

            }
            conn = new OleDbConnection(connectionString);
            try
            {
                conn.Open(); //데이터베이스 연결
                OleDbCommand cmd = new OleDbCommand();
                cmd.CommandText = "INSERT INTO MEMBER VALUES('" + txtID.Text + "','" + txtName.Text + "','" + txtPW.Text + "')";

                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn;

                cmd.ExecuteNonQuery(); //쿼리문을 실행하고 영향받는 행의 수를 반환.
                MessageBox.Show("가입이 완료되었습니다");

            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message); //에러 메세지 
            }
            finally
            {
                if (conn != null)
                {
                    conn.Close(); //데이터베이스 연결 해제
                }
            }

        }

        private void button8_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void txtID_TextChanged(object sender, EventArgs e)
        {
            idcheck = false;
        }
    }
}
