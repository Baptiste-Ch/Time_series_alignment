"""Add to VerifData

Revision ID: 470eb904b397
Revises: 38c43395ea72
Create Date: 2023-08-22 12:31:38.104718

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '470eb904b397'
down_revision = '38c43395ea72'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('dropdown_data')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('dropdown_data',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('dropdown_value', sa.VARCHAR(length=100), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###
