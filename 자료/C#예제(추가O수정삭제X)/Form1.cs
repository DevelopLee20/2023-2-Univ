using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Data.OleDb;

namespace WindowsFormsApplication1
{
    public partial class Form1 : Form
    {
        OleDbConnection conn;
        string connectionString;

        
        public Form1()
        {
            InitializeComponent();
            dataGridView1.Rows.Clear();


        }
        private void button1_Click(object sender, EventArgs e)
        {

            connectionString = "Provider=MSDAORA;Password=" + txtBoxPw.Text + ";User ID=" + txtboxId.Text;//oracle 서버 연결

            //연결 스트링에 대한 정보 
            //Oracle - MSDAORA 

            conn = new OleDbConnection(connectionString);
            conn.Open(); //데이터베이스 연결
            updatedb();
 
        }
        private void updatedb()
        {
            try
            {
                //conn.Open(); //데이터베이스 연결
                OleDbCommand cmd = new OleDbCommand();
                cmd.CommandText = "select * from emp"; //member 테이블
                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn;

                OleDbDataReader read = cmd.ExecuteReader(); //select * from emp 결과
                dataGridView1.ColumnCount = 9;
                //필드명 받아오는 반복문
                for (int i = 0; i < 8; i++)
                {
                    dataGridView1.Columns[i].Name = read.GetName(i);
                }

                //행 단위로 반복
                while (read.Read())
                {
                    object[] obj = new object[8]; // 필드수만큼 오브젝트 배열

                    for (int i = 0; i < 8; i++) // 필드 수만큼 반복
                    {
                        obj[i] = new object();
                        obj[i] = read.GetValue(i); // 오브젝트배열에 데이터 저장
                    }
 
                    dataGridView1.Rows.Add(obj); //데이터그리드뷰에 오브젝트 배열 추가
                }

                read.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message); //에러 메세지 
            }
        }

        
        private void dataGridView1_CellContentClick(object sender, DataGridViewCellEventArgs e) //그리드뷰의 셀을 클릭했을때
         {
             

               txt0.Text = dataGridView1.Rows[e.RowIndex].Cells[0].Value.ToString();
               txt1.Text = dataGridView1.Rows[e.RowIndex].Cells[1].Value.ToString();
               txt2.Text = dataGridView1.Rows[e.RowIndex].Cells[2].Value.ToString();
               txt3.Text = dataGridView1.Rows[e.RowIndex].Cells[3].Value.ToString();
               txt4.Text = dataGridView1.Rows[e.RowIndex].Cells[4].Value.ToString().Substring(0,10);
               txt5.Text = dataGridView1.Rows[e.RowIndex].Cells[5].Value.ToString();
               txt6.Text = dataGridView1.Rows[e.RowIndex].Cells[6].Value.ToString();
               txt7.Text = dataGridView1.Rows[e.RowIndex].Cells[7].Value.ToString();

             
         }
        

        private void button2_Click(object sender, EventArgs e) //추가버튼
        {
         dataGridView1.Rows.Clear();


         conn = new OleDbConnection(connectionString);
            try
            {
                conn.Open(); //데이터베이스 연결
                OleDbCommand cmd = new OleDbCommand();
                if (txt6.Text == "")
                {
                    cmd.CommandText = "INSERT INTO emp VALUES(" + txt0.Text + ",'" + txt1.Text + "','" + txt2.Text + "'," + txt3.Text + ",'" + txt4.Text + "'," + txt5.Text + "," +"NULL," + txt7.Text + ")";  //member 테이블
                }
                else
                {
                    cmd.CommandText = "INSERT INTO emp VALUES(" + txt0.Text + ",'" + txt1.Text + "','" + txt2.Text + "'," + txt3.Text + ",'" + txt4.Text + "'," + txt5.Text + "," + txt6.Text + "," + txt7.Text + ")";  //member 테이블
                }
                textBox1.Text = cmd.CommandText;
                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn;

                cmd.ExecuteNonQuery(); //쿼리문을 실행하고 영향받는 행의 수를 반환.
                updatedb();

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

        private void button3_Click(object sender, EventArgs e) //삭제버튼
        {
        dataGridView1.Rows.Clear();

        conn = new OleDbConnection(connectionString);
            try
            {






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

        private void button4_Click(object sender, EventArgs e) //수정버튼
        {
            dataGridView1.Rows.Clear();

            conn = new OleDbConnection(connectionString);
            try
            {








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

        private void button5_Click(object sender, EventArgs e) //초기화버튼
        {
            dataGridView1.Rows.Clear();
            txt0.Clear();
            txt1.Clear();
            txt2.Clear();
            txt3.Clear();
            txt4.Clear();
            txt5.Clear();
            txt6.Clear();
            txt7.Clear();
            conn = new OleDbConnection(connectionString);
            conn.Open(); //데이터베이스 연결
            updatedb();
        }

        private void txtBoxPw_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
            {
                button1_Click(sender, e);
            }

        }
    }
}

      